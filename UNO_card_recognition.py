import cv2 as cv
import numpy as np
import os

class Card:
    def __init__(self, image, box, center):
        self.image = image
        self.box = box
        self.center = center
        self.color = None
        self.symbol = None

class CardDetector:
    def __init__(self, image_path):
        self.image = cv.imread(image_path)
        self.image_scaled = cv.resize(self.image, (0, 0), fx=0.6, fy=0.6)
        self.cards = []

    # Filtracja w celu uwydatnienia krawędzi kart
    def filtering_edges_card(self, image):
        img_median = cv.medianBlur(image, 13)  # Usuwanie szumów za pomocą rozmycia medianowego
        img_gaussian = cv.GaussianBlur(img_median, (15, 15), 15.0)  # Dodatkowe rozmycie za pomocą filtra Gaussa
        img_unsharped = cv.addWeighted(img_median, 2.0, img_gaussian, -1.0, 0)  # Wyostrzenie obrazu
        img_gray = cv.cvtColor(img_unsharped, cv.COLOR_BGR2GRAY)  # Konwersja obrazu do skali szarości
        img_inverted = cv.bitwise_not(img_gray)  # Inwersja obrazu
        _, img_binary = cv.threshold(img_inverted, 230, 255, cv.THRESH_BINARY)  # Próg binarny
        img_binary_inv = cv.bitwise_not(img_binary)  # Inwersja obrazu
        kernel = np.ones((8, 8), np.uint8)  # Kernel do erozji
        eroded_image = cv.erode(img_binary_inv, kernel, iterations=1)  # Erozja
        canny = cv.Canny(eroded_image, 120, 255, 1)  # Detekcja krawędzi Canny'ego
        return canny

    # Funkcja wycinająca kartę
    def cut_cards(self):
        # Filtracja zdjęcia w celu uwydatnienia konturów
        filtred_img = self.filtering_edges_card(self.image_scaled)
        contours, _ = cv.findContours(filtred_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Opisanie prostokąta wokół każdej karty
        for cnt, outline in enumerate(contours):
            min_rectangle = cv.minAreaRect(outline)
            box = cv.boxPoints(min_rectangle)
            box = np.intp(box)

            # Rysowanie prostokąta na obrazie
            cv.polylines(self.image_scaled, [box], -1, (0, 255, 0), 1)

            # Sortowanie wierzchołków w odpowiedniej kolejności
            sorted_box = sorted(box, key=lambda coord: coord[0] + coord[1]) 
            x2, _ = sorted_box[1]
            x3, _ = sorted_box[2]
            if x3 > x2:
                sorted_box[1], sorted_box[2] = sorted_box[2], sorted_box[1]

            # Wycięcie karty w formacie 210x300 px
            width, height = 210, 300
            points_first = np.float32(sorted_box)
            points_after = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            transform_matrix = cv.getPerspectiveTransform(points_first, points_after)
            card_image = cv.warpPerspective(self.image_scaled, transform_matrix, (width, height))

            # Tworzenie obiektu karty
            card = Card(card_image, sorted_box, np.intp(min_rectangle[0]))
            self.cards.append(card)

    def filtering_edges_number(self, card_image):
        card_gray = cv.cvtColor(card_image, cv.COLOR_BGR2GRAY)
        card_median = cv.medianBlur(card_gray, 15)
        card_gaussian = cv.GaussianBlur(card_median, (11, 11), 5)
        card_gaussian_inv = cv.bitwise_not(card_gaussian)
        cv.equalizeHist(card_gaussian_inv, card_gaussian_inv)
        card_gaussian_inv = cv.bitwise_not(card_gaussian_inv)

        # Zastosowanie progowania adaptacyjnego do uzyskania binarnego obrazu
        card_binary = cv.adaptiveThreshold(card_gaussian_inv, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 109, -70)

        # Erozja obrazu
        kernel = np.ones((3, 3), np.uint8)
        card_eroded = cv.erode(card_binary, kernel, iterations=1)

        # Dylatacja obrazu
        kernel = np.ones((7, 7), np.uint8)
        card_dilated = cv.dilate(card_eroded, kernel, iterations=1)

        # Użycie detektora krawędzi Canny'ego
        canny = cv.Canny(card_dilated, 120, 255, 1)
        return canny

    def masking_card_elipse(self, image_card, card_clean):
        height, width = card_clean.shape[:2]
        center_coordinates = (width // 2, height // 2)
        mask = np.zeros((height, width), np.uint8)
        axeslength = (90, 55)

        # Tworzenie maski w kształcie elipsy
        cv.ellipse(mask, center_coordinates, axeslength, -60, 0, 360, (255, 255, 255), -1)

        # Nakładanie maski na obraz
        masked_card = cv.bitwise_and(image_card, image_card, mask=mask)
        return masked_card

    def masking_card_rectangle(self, image_card, card_clean):
        height, width = card_clean.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        start_poit = (60, 90)
        end_point = (150, 200)

        # Tworzenie maski w kształcie prostokąta
        cv.rectangle(mask, start_poit, end_point, (255, 255, 255), -1)

        # Nakładanie maski na obraz
        masked_card = cv.bitwise_and(image_card, image_card, mask=mask)
        return masked_card

    def get_hu_moments(self, img_card):
        moments = cv.moments(img_card)
        hu_moments = cv.HuMoments(moments)
        return hu_moments

    def get_symbol(self, hu_moments):
        # zakres momentów dla 1
        lower_moment_hu_0_1 = 1.355e-02
        upper_moment_hu_0_1 = 1.881e-02
        lower_moment_hu_2_1 = 1.64e-07
        upper_moment_hu_2_1 = 6.2e-07

        # zakres momentów dla 5
        lower_moment_hu_0_5 = 1.32e-02
        upper_moment_hu_0_5 = 1.2232e-02
        lower_moment_hu_1_5 = 1.28e-05
        upper_moment_hu_1_5 = 1.95e-05

        # zakres momentów dla 3
        lower_moment_hu_0_3 = 1.225e-02
        upper_moment_hu_0_3 = 1.31e-02
        lower_moment_hu_1_3 = 1.44e-05
        upper_moment_hu_1_3 = 2.2856e-05

        # zakres momentów dla zmiany kolejki
        lower_moment_hu_0_zk = 1.22e-02
        upper_moment_hu_0_zk = 1.805e-02
        lower_moment_hu_1_zk = 0.76e-04
        upper_moment_hu_1_zk = 1.377e-04

        # zakres momentów dla (+2)
        lower_moment_hu_0_plus_2 = 2.11e-02
        upper_moment_hu_0_plus_2 = 2.279e-02
        lower_moment_hu_1_plus_2 = 0.939e-04
        upper_moment_hu_1_plus_2 = 1.1e-04

        # Sprawdzenie znaku
        if (hu_moments[0] > lower_moment_hu_0_plus_2 and hu_moments[0] < upper_moment_hu_0_plus_2 and
                hu_moments[1] > lower_moment_hu_1_plus_2 and hu_moments[1] < upper_moment_hu_1_plus_2):
            return "+2"
        elif (hu_moments[0] > lower_moment_hu_0_1 and hu_moments[0] < upper_moment_hu_0_1 and
              hu_moments[2] > lower_moment_hu_2_1 and hu_moments[2] < upper_moment_hu_2_1):
            return "1"
        elif (hu_moments[0] > lower_moment_hu_0_zk and hu_moments[0] < upper_moment_hu_0_zk and
              hu_moments[1] > lower_moment_hu_1_zk and hu_moments[1] < upper_moment_hu_1_zk):
            return "ZK"
        elif (hu_moments[0] > lower_moment_hu_0_3 and hu_moments[0] < upper_moment_hu_0_3 and
              hu_moments[1] > lower_moment_hu_1_3 and hu_moments[1] < upper_moment_hu_1_3):
            return "3"
        elif ((hu_moments[0] > lower_moment_hu_0_5 or hu_moments[0] < upper_moment_hu_0_5) and
              hu_moments[1] > lower_moment_hu_1_5 and hu_moments[1] < upper_moment_hu_1_5):
            return "5"
        else:
            return ":("

    # Funkcja wyznaczająca średni kolor wycinka karty w HSV
    def color_average(self, img):
        hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        img_clipping = hsv_img[180:200, 20:50]
        color = cv.mean(img_clipping)
        return color

    # Okteślenie koloru karty
    def get_color(self, pixel_card):
        lower_green = np.array([45, 30, 30])
        upper_green = np.array([65, 255, 255])
        lower_blue = np.array([1, 30, 30])
        upper_blue = np.array([25, 255, 255])
        lower_yellow = np.array([80, 30, 30])
        upper_yellow = np.array([100, 255, 255])
        lower_red = np.array([110, 30, 30])
        upper_red = np.array([125, 255, 255])

        # Warunek na znajdowanie się wartości piksela w granicach poszczególnych kolorów
        if pixel_card[0] >= lower_green[0] and pixel_card[0] <= upper_green[0]:
            return "ZIELONA"
        if pixel_card[0] >= lower_blue[0] and pixel_card[0] <= upper_blue[0]:
            return "NIEBIESKA"
        if pixel_card[0] >= lower_yellow[0] and pixel_card[0] <= upper_yellow[0]:
            return "ZOLTA"
        if pixel_card[0] >= lower_red[0] and pixel_card[0] <= upper_red[0]:
            return "CZERWONA"

    def process_cards(self):
        for card in self.cards:
            binary_card = self.filtering_edges_number(card.image)
            masked_binary_card = self.masking_card_elipse(binary_card, card.image)
            masked_binary_card = self.masking_card_rectangle(masked_binary_card, card.image)
            hu_moments = self.get_hu_moments(masked_binary_card)
            card.symbol = self.get_symbol(hu_moments)
            card.color = self.get_color(self.color_average(card.image))

    def show_cards(self): 
        for card in self.cards:
            # Przypisanie współżędnych lewego górnego narożnika
            box = card.box[0]
            box[1] -= 10
            center = str(card.center)

            # Sprawdzenie warunku czy karta jest specjalna i ewentualne wyświetlenie 
            if card.symbol == "ZK" or card.symbol == "+2":
                cv.putText(self.image_scaled, center, tuple(box), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))

            box[1] -= 40
            text = f"{card.color}  {card.symbol}"
            cv.putText(self.image_scaled, text, tuple(box), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))

        cv.imshow("Zdjecie kart z opisami", self.image_scaled)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_single_card(self):
        for index ,card in enumerate(self.cards):
            cv.imshow(f"Karta{index+1}" ,card.image)
            cv.waitKey(0)
            cv.destroyAllWindows()

class ImageMenu:
    def __init__(self, directory):
        self.directory = directory
        self.images = self.get_image_files()

    def get_image_files(self):
        """Funkcja zwraca listę plików obrazów w katalogu"""
        return [f for f in os.listdir(self.directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    def display_menu(self):
        """Wyświetla menu wyboru pliku"""
        print(f"\nWybierz plik, który chcesz otworzyć (1-{len(self.images)}):")
        for idx, image in enumerate(self.images, 1):
            print(f"{idx}. {image}")
        print("q. Wyjście")

    def get_user_choice(self):
        """Pobiera wybór użytkownika"""
        while True:
            choice = input(f"Wprowadź numer pliku lub opcję (1-{len(self.images)}, q): ")
            
            if choice == 'q':
                return choice  # Opcja 'q' do wyjścia
            try:
                choice = int(choice)
                if 1 <= choice <= len(self.images):
                    return choice - 1  # Zwracamy indeks pliku (0-indexed)
                else:
                    print(f"Niepoprawny numer, proszę wybrać numer od 1 do {len(self.images)} lub 'q' na wyjście.")
            except ValueError:
                print("Proszę podać liczbę lub 'q'.")

    def display_action_menu(self):
        """Wyświetla menu wyboru akcji (co zrobić z wybranym obrazem)"""
        print("\nCo chcesz zrobić z wybranym obrazem?")
        print("1. Zaznacz karty na zdjęciu")
        print("2. Wyświetl tylko karty")
        print("3. Powrót do menu głównego")

    def get_action_choice(self):
        """Pobiera wybór akcji od użytkownika"""
        while True:
            try:
                action = int(input("Wprowadź numer akcji (1-3): "))
                if action == 1 or action == 2:
                    return action
                elif action == 3:
                    return action  # Opcja 3 (powrót do menu głównego)
                else:
                    print("Niepoprawny wybór. Wybierz 1, 2 lub 3.")
            except ValueError:
                print("Proszę podać liczbę (1, 2 lub 3).")

    def run(self):
        """Uruchamia menu"""
        while True:
            self.display_menu()
            choice = self.get_user_choice()

            if choice == 'q':
                print("Zakończenie programu.")
                break  # Kończy działanie programu

            selected_image = self.images[choice]
            print(f"Wybrano: {selected_image}")

            # Pobieramy wybór akcji
            self.display_action_menu()
            action_choice = self.get_action_choice()

            if action_choice == 3:
                continue  # Powrót do menu głównego

            # Przekazujemy wybrany obrazek do detekcji kart
            detector = CardDetector(os.path.join(self.directory, selected_image))
            
            if action_choice == 1:
                # Zaznacz karty na zdjęciu
                detector.cut_cards()
                detector.process_cards()
                detector.show_cards()
            elif action_choice == 2:
                # Wyświetl tylko karty
                detector.cut_cards()
                detector.process_cards()
                detector.show_single_card()


if __name__ == "__main__":
    menu = ImageMenu('./Images')
    menu.run()