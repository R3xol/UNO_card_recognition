import pytest
from unittest.mock import MagicMock
from CardDetector import CardDetector
from CardDetector import Card
import cv2 as cv
import numpy as np

import tempfile

def test_card_detector_init():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        blank_image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv.imwrite(temp_img.name, blank_image)

        detector = CardDetector(image_path=temp_img.name)

        assert detector.image is not None
        assert detector.image_scaled is not None
        assert detector.cards == []  # Powinna być pusta lista kart


def test_process_cards_calls_methods():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        # Tworzymy czarny obraz (100x100 px)
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv.imwrite(temp_img.name, blank_image)

        detector = CardDetector(image_path=temp_img.name)
        assert detector.image is not None

    # Tworzymy atrapę obiektu karty
    mock_card = MagicMock()
    mock_card.image = "fake_image"
    detector.cards = [mock_card]

    # Podmieniamy metody na mocki
    detector._filtering_edges_number = MagicMock()
    detector._masking_card_elipse = MagicMock(return_value="masked_image")
    detector._masking_card_rectangle = MagicMock(return_value="masked_image")
    detector._get_hu_moments = MagicMock(return_value="hu_moments")
    detector._get_symbol = MagicMock(return_value="symbol")
    detector._get_color = MagicMock(return_value="color")
    detector.color_average = MagicMock(return_value="average_color")

    # Uruchamiamy testowaną metodę
    detector.process_cards()

    # Sprawdzamy, czy każda metoda została wywołana
    detector._filtering_edges_number.assert_called_once_with(mock_card.image)
    detector._masking_card_elipse.assert_called_once()
    detector._masking_card_rectangle.assert_called_once()
    detector._get_hu_moments.assert_called_once()
    detector._get_symbol.assert_called_once()
    detector._get_color.assert_called_once()
    detector.color_average.assert_called_once_with(mock_card.image)

def test_cut_cards():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        blank_image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv.rectangle(blank_image, (100, 100), (300, 300), (255, 255, 255), -1)  # Dodajemy biały kwadrat jako karta
        cv.imwrite(temp_img.name, blank_image)

        detector = CardDetector(image_path=temp_img.name)

        detector.cut_cards()

        assert len(detector.cards) == 1
        assert isinstance(detector.cards[0].image, np.ndarray)  # Obraz karty


def test_get_symbol():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv.imwrite(temp_img.name, blank_image)
        
        detector = CardDetector(image_path=temp_img.name)

    assert detector._get_symbol([1.4e-02, 1.7e-07, 2e-07]) == "1"


def test_get_color():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv.imwrite(temp_img.name, blank_image)

        detector = CardDetector(image_path=temp_img.name)

        assert detector._get_color(np.array([50, 200, 200])) == "ZIELONA"
        assert detector._get_color(np.array([10, 200, 200])) == "NIEBIESKA"
        assert detector._get_color(np.array([90, 200, 200])) == "ZOLTA"
        assert detector._get_color(np.array([120, 200, 200])) == "CZERWONA"


def test_show_cards():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        blank_image = np.zeros((1200, 1600, 3), dtype=np.uint8)
        vertices = np.array([[92/0.7, 254/0.7], [269/0.7, 256/0.7], [266/0.7, 527/0.7], [90/0.7, 525/0.7]], dtype=np.int32)
        cv.polylines(blank_image, [vertices], isClosed=True, color=(255, 255, 255), thickness=2)
        cv.imwrite(temp_img.name, blank_image)

        detector = CardDetector(image_path=temp_img.name)
        detector.cards = [Card(np.zeros((210, 300, 3), dtype=np.uint8), [[ 92, 254], [269, 256], [ 90, 525], [266, 527]], [179, 391], "ZIELONA", "5")]

        detector.show_cards()  # Upewnij się, że działa bez wyjątków 


def test_number_of_detected_cards():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        blank_image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv.rectangle(blank_image, (50, 50), (150, 200), (255, 255, 255), -1)
        cv.rectangle(blank_image, (200, 200), (350, 300), (255, 255, 255), -1)
        cv.imwrite(temp_img.name, blank_image)

        detector = CardDetector(image_path=temp_img.name)

        detector.cut_cards()

        assert len(detector.cards) == 2
