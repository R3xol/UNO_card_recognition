from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Union

@dataclass
class Card:
    image: np.ndarray                       # Macierz przechowujaca karte (RGB)
    box: List[int]                          # Lista 4 liczb (np. [x1, y1, x2, y2])
    center: np.ndarray  # Środek jako macierz lub lista współrzędnych [x, y]
    color: Optional[str] = None             # Kolor, np. "czerwony", domyślnie brak
    symbol: Optional[str] = None            # Symbol (np. "1"), domyślnie brak
