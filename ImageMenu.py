from CardDetector import CardDetector
import os
import cv2 as cv

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
                    print(f"Niepoprawny numer, proszę wybrać numer od 1 do {len(self.images)} lub 'q' aby wyjście.")
            except ValueError:
                print("Proszę podać liczbę całkowitą lub 'q'.")

    def display_action_menu(self):
        """Wyświetla menu wyboru akcji (co zrobić z wybranym obrazem)"""
        print("\nCo chcesz zrobić z wybranym obrazem?")
        print("1. Zaznacz karty na zdjęciu")
        print("2. Wyświetl karty")
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
                print("Zakończenie działania programu.")
                break  # Kończy działanie programu

            selected_image = self.images[choice]
            print(f"Wybrano: {selected_image}")

            # Pobieramy wybór akcji
            self.display_action_menu()
            action_choice = self.get_action_choice()

            if action_choice == 3:
                continue  # Powrót do menu głównego

            # Przekazujemy wybrany obrazek do detekcji kart
            image = cv.imread(os.path.join(self.directory, selected_image))
            detector = CardDetector(image)
            
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