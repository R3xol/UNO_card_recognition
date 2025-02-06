import cv2 as cv
import numpy as np

# Filtracja w celu uwydatnienia krawędzi kart
def filtering_edges_card(image):
    img_median=cv.medianBlur(image, 13) 
    img_gaussian = cv.GaussianBlur(img_median, (15, 15), 15.0) 
    img_unsharped = cv.addWeighted(img_median, 2.0, img_gaussian, -1.0, 0)
    img_gray = cv.cvtColor(img_unsharped, cv.COLOR_BGR2GRAY)
    img_inverted=cv.bitwise_not(img_gray) 
    _, img_binary = cv.threshold(img_inverted, 230, 255, cv.THRESH_BINARY) 
    img_binary_inv=cv.bitwise_not(img_binary) 
    kernel = np.ones((8,8), np.uint8)
    eroded_image = cv.erode(img_binary_inv, kernel, iterations=1)
    canny = cv.Canny(eroded_image, 120, 255, 1)
    return canny

# Funkcja wycinająca kartę
def cut_cards(image):
    # Filtracja zdjęcia w celu uwydatnienia konturów
    filtred_img = filtering_edges_card(image)
    contours, _ = cv.findContours(filtred_img , cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Opisanie prostokąta na karcie
    for cnt, outline in enumerate(contours):
        min_rectangle = cv.minAreaRect(outline)
        box = cv.boxPoints(min_rectangle)
        box = np.intp(box)
        cv.polylines(image, [box], -1, (0, 255, 0), 1)
        #Uszeregowanie wierzchołków karty
        sorted_box = sorted(box, key=lambda coord: coord[0] + coord[1])
        x2,_ = sorted_box[1]
        x3,_ = sorted_box[2]
        if x3>x2:
            pom=sorted_box[1]
            sorted_box[1]=sorted_box[2]
            sorted_box[2]=pom
        # Wycięcie karty
        width, height = 210, 300
        points_first = np.float32(sorted_box)
        points_after = np.float32([[0, 0], [width, 0], [0, height],[width, height]])
        transform_matrix = cv.getPerspectiveTransform(points_first, points_after)
        card = cv.warpPerspective(image, transform_matrix, (width, height))
        # Przypisanie poszczególnych kart od zmiennych
        if cnt == 0:
            card_1 = card
            box_1 = sorted_box
        elif cnt == 1:
            card_2 = card
            box_2 = sorted_box
        elif cnt == 2:
            card_3 = card
            box_3 = sorted_box
        elif cnt == 3:
            card_4 = card
            box_4 = sorted_box
    return card_1, card_2, card_3, card_4, box_1, box_2, box_3, box_4

# Funkcja pozostawiająca jedynie binarny kontur karty
def filtering_edges_number(card):
    card_gray = cv.cvtColor(card, cv.COLOR_BGR2GRAY)
    card_median = cv.medianBlur(card_gray, 15)
    card_gaussian = cv.GaussianBlur(card_median, (11,11), 5)
    card = card_gaussian
    card_gaussian_inv = cv.bitwise_not(card_gaussian)
    cv.equalizeHist(card_gaussian_inv, card)
    card = cv.bitwise_not(card)
    card_binary = cv.adaptiveThreshold(card, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 109, -70)
    kernel = np.ones((3,3), np.uint8)
    card_eroded = cv.erode(card_binary , kernel, iterations=1)
    kernel = np.ones((7,7), np.uint8)
    card_dilated = cv.dilate(card_eroded, kernel, iterations=1)
    canny = cv.Canny(card_dilated , 120, 255, 1)
    return canny

# Funkcja maskująca wszystko poza fragmentem karty z symbolem (elipsa na środku)
def masking_card_elipse(image_card,card_clean):
    height, width = card_clean.shape[:2]
    center_coordinates = (width // 2, height // 2)
    mask = np.zeros((height, width), np.uint8)
    axeslength=(90, 55)
    cv.ellipse(mask, center_coordinates, axeslength, -60, 0, 360, (255,255,255), -1)
    masked_card = cv.bitwise_and(image_card, image_card, mask=mask)
    return masked_card

# Funkcja maskująca wszystko poza fragmentem karty z symbolem (prostokąt na środku)
def masking_card_rectangle(image_card,card_clean):
    height, width = card_clean.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    start_poit = (60,90)
    end_point = (150,200)
    cv.rectangle(mask, start_poit, end_point, (255,255,255), -1)
    masked_card = cv.bitwise_and(image_card, image_card, mask=mask)
    return masked_card

# Wyznaczenie hu momentów karty
def get_hu_moments(img_card):
    moments=cv.moments(img_card)
    hu_moments=cv.HuMoments(moments)
    return hu_moments

# Funkcja sprawdzająca na podstawie momentów znak karty
def get_symbol(hu_moments):
    # zakres momentów dla 1
    lower_moment_hu_0_1=1.355e-02
    upper_moment_hu_0_1=1.881e-02
    lower_moment_hu_2_1=1.64e-07 
    upper_moment_hu_2_1=6.2e-07

    # zakres momentów dla 5
    lower_moment_hu_0_5=1.32e-02
    upper_moment_hu_0_5=1.2232e-02
    lower_moment_hu_1_5=1.28e-05
    upper_moment_hu_1_5=1.95e-05

    # zakres momentów dla 3
    lower_moment_hu_0_3=1.225e-02
    upper_moment_hu_0_3=1.31e-02
    lower_moment_hu_1_3=1.44e-05
    upper_moment_hu_1_3=2.2856e-05

    # zakres momentów dla zmiany kolejki
    lower_moment_hu_0_zk=1.22e-02
    upper_moment_hu_0_zk=1.805e-02
    lower_moment_hu_1_zk=0.76e-04
    upper_moment_hu_1_zk=1.377e-04

    # zakres momentów dla (+2)
    lower_moment_hu_0_plus_2=2.11e-02
    upper_moment_hu_0_plus_2=2.279e-02
    lower_moment_hu_1_plus_2=0.939e-04
    upper_moment_hu_1_plus_2=1.1e-04
    
    # Sprawdzenie warunku na +2
    if(hu_moments[0]>lower_moment_hu_0_plus_2 and hu_moments[0]<upper_moment_hu_0_plus_2 and hu_moments[1]>lower_moment_hu_1_plus_2 and hu_moments[1]<upper_moment_hu_1_plus_2):
        return "+2"
    # Sprawdzenie warunku na 1
    elif(hu_moments[0]>lower_moment_hu_0_1 and hu_moments[0]<upper_moment_hu_0_1 and hu_moments[2]>lower_moment_hu_2_1 and hu_moments[2]<upper_moment_hu_2_1):
        return "1"
    # Sprawdzenie warunku na zmiane kolejki
    elif(hu_moments[0]>lower_moment_hu_0_zk and hu_moments[0]<upper_moment_hu_0_zk and hu_moments[1]>lower_moment_hu_1_zk and hu_moments[1]<upper_moment_hu_1_zk):
        return "ZK"
    # Sprawdzenie warunku na 3
    elif(hu_moments[0]>lower_moment_hu_0_3 and hu_moments[0]<upper_moment_hu_0_3 and hu_moments[1]>lower_moment_hu_1_3 and hu_moments[1]<upper_moment_hu_1_3):
        return "3"
    # Sprawdzenie warunku na 5
    elif((hu_moments[0]>lower_moment_hu_0_5 or hu_moments[0]<upper_moment_hu_0_5) and hu_moments[1]>lower_moment_hu_1_5 and hu_moments[1]<upper_moment_hu_1_5):
        return "5"
    # Gdyby karta nie została rozpoznana     
    else:
        return ":("
    
# Funkcja wyznaczająca średni kolor wycinka karty w HSV
def color_average(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    img_clipping = hsv_img[180:200,20:50]
    color = cv.mean(img_clipping)
    return color

# Okteślenie koloru karty
def get_color(pixel_card):
    # Zakresy wartości dla niebieskiego
    lower_blue=np.array([1, 30, 30])
    upper_blue=np.array([25, 255, 255])
    # Zakresy wartości dla zielonego
    lower_green=np.array([45, 30, 30])
    upper_green=np.array([65, 255,255])
    # Zakresy wartości dla żółtego
    lower_yellow=np.array([80, 30, 30])
    upper_yellow=np.array([100, 255, 255])
    # Zakresy wartości dla czerwonego
    lower_red=np.array([110, 30, 30])
    upper_red=np.array([125, 255, 255])

    #warunek na znajdowanie się wartości piksela w granicach poszczególnych kolorów
    if pixel_card[0]>=lower_green[0] and pixel_card[0]<=upper_green[0]:
        return "ZIELONA"
    if pixel_card[0]>=lower_blue[0] and pixel_card[0]<=upper_blue[0]:
        return "NIEBIESKA"
    if pixel_card[0]>=lower_yellow[0] and pixel_card[0]<=upper_yellow[0]:
        return "ZOLTA"
    if pixel_card[0]>=lower_red[0] and pixel_card[0]<=upper_red[0]:
        return "CZERWONA"

# Pobranie koordynatów środków karty
def center_coordinates(image):
    filtred_img = filtering_edges_card(image)
    contours, _ = cv.findContours(filtred_img , cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Opisanie prostokąta na karcie
    for cnt, outline in enumerate(contours):
        min_rectangle = cv.minAreaRect(outline)
        if cnt == 0:
            center_card_1 = np.intp(min_rectangle[0])
        elif cnt == 1:
            center_card_2 = np.intp(min_rectangle[0])
        elif cnt == 2:
            center_card_3 = np.intp(min_rectangle[0])
        elif cnt == 3:
            center_card_4 = np.intp(min_rectangle[0])
    return center_card_1, center_card_2, center_card_3, center_card_4

# Wyświettenie wejściowego obrazka z podpisanymi kartami
def show_card(box_1, box_2, box_3, box_4, center_card_1, center_card_2, center_card_3, center_card_4, 
    color_card_1 , color_card_2, color_card_3, color_card_4, symbol_card_1, symbol_card_2, symbol_card_3, symbol_card_4):
    # Przypisanie współżędnych lewego górnego narożnika
    box_1 = box_1[0] 
    box_2 = box_2[0]
    box_3 = box_3[0]
    box_4 = box_4[0]

    box_1[1] = box_1[1] - 10
    box_2[1] = box_2[1] - 10
    box_3[1] = box_3[1] - 10
    box_4[1] = box_4[1] - 10
    # Żótowanie tablicy int na string
    center_card_1 = str(center_card_1)
    center_card_2 = str(center_card_2)
    center_card_3 = str(center_card_3)
    center_card_4 = str(center_card_4)
    # Sprawdzenie warunku czy karta jest specjalna i ewentualne wyświetlenie 
    if symbol_card_1 == "ZK" or symbol_card_1 == "+2":
        cv.putText(img_scaled, center_card_1, (box_1), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))

    if symbol_card_2 == "ZK" or symbol_card_2 == "+2":
        cv.putText(img_scaled, center_card_2, (box_2), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))

    if symbol_card_3 == "ZK" or symbol_card_3 == "+2":
        cv.putText(img_scaled, center_card_3, (box_3), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))

    if symbol_card_4 == "ZK" or symbol_card_4 == "+2":
        cv.putText(img_scaled, center_card_4, (box_4), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))

    box_1[1] = box_1[1] - 40
    box_2[1] = box_2[1] - 40
    box_3[1] = box_3[1] - 40
    box_4[1] = box_4[1] - 40

    text_card_1 = color_card_1 + "  " + symbol_card_1 
    text_card_2 = color_card_2 + "  " + symbol_card_2
    text_card_3 = color_card_3 + "  " + symbol_card_3
    text_card_4 = color_card_4 + "  " + symbol_card_4

    cv.putText(img_scaled, text_card_1, (box_1), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
    cv.putText(img_scaled, text_card_2, (box_2), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
    cv.putText(img_scaled, text_card_3, (box_3), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
    cv.putText(img_scaled, text_card_4, (box_4), cv.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))

    return img_scaled

# Wczytanie zdjęcia
img = cv.imread('1.png')

# Przeskalowanie, aby łatwiej śledzić zmiany na małym ekranie komputera:(
img_scaled = cv.resize(img, (0, 0), fx=0.6, fy=0.6)

# Przypisanie wyciętych kart oraz współżędnych wieszchołków do zmiennych
card_1, card_2, card_3, card_4, box_1, box_2, box_3, box_4 = cut_cards(img_scaled)

# Filtracja w celu pozostwienia jedynie krawędzi na karcie (symbol)
binary_card_1 = filtering_edges_number(card_1)
binary_card_2 = filtering_edges_number(card_2)
binary_card_3 = filtering_edges_number(card_3)
binary_card_4 = filtering_edges_number(card_4)

# Zamaskowanie wszystkiego oprócz interesującego kawałka karty poprzez podujne maskowanie
masked_binary_card_1 = masking_card_elipse(binary_card_1, card_1)
masked_binary_card_2 = masking_card_elipse(binary_card_2, card_2)
masked_binary_card_3 = masking_card_elipse(binary_card_3, card_3)
masked_binary_card_4 = masking_card_elipse(binary_card_4, card_4)

masked_binary_card_1 = masking_card_rectangle(masked_binary_card_1, card_1)
masked_binary_card_2 = masking_card_rectangle(masked_binary_card_2, card_2)
masked_binary_card_3 = masking_card_rectangle(masked_binary_card_3, card_3)
masked_binary_card_4 = masking_card_rectangle(masked_binary_card_4, card_4)

# Wyznaczenie wartosci momentów hu
moment_hu_1 = get_hu_moments(masked_binary_card_1)
moment_hu_2 = get_hu_moments(masked_binary_card_2)
moment_hu_3 = get_hu_moments(masked_binary_card_3)
moment_hu_4 = get_hu_moments(masked_binary_card_4)

# Odczytanie synmbolu karty na podstawie momentów
symbol_card_1 = get_symbol(moment_hu_1)
symbol_card_2 = get_symbol(moment_hu_2)
symbol_card_3 = get_symbol(moment_hu_3)
symbol_card_4 = get_symbol(moment_hu_4)

# Odczytanie koloru karty na podstawie odcieni w HSV
color_card_1 = get_color(color_average(card_1))
color_card_2 = get_color(color_average(card_2))
color_card_3 = get_color(color_average(card_3))
color_card_4 = get_color(color_average(card_4))

# Wyznaczenie koordynatów środka każdej karty (możemy wyznaczyć je z przeskalowanej lub startowj karty)
center_card_1, center_card_2, center_card_3, center_card_4 = center_coordinates(img)

# Wyświetlenie wszystkiego na wejściowym obrazku (już przeskalowanym)
final_image = show_card(box_1, box_2, box_3, box_4, center_card_1, center_card_2, center_card_3, center_card_4, 
color_card_1 , color_card_2, color_card_3, color_card_4, symbol_card_1, symbol_card_2, symbol_card_3, symbol_card_4)

# Wyświetlenie opisanego zdjęcia
cv.imshow("Zdjecie kart z opisami",final_image)

cv.waitKey(0)
cv.destroyAllWindows()