from typing import Generator
from PIL import Image
from numpy import array, uint8, ndarray
from cv2 import resize, INTER_AREA

def notenames() -> list[str]:
    return [
        f"{note}{octave}"
        for octave in range(1,9)
        for note in "CDEFGAB"
    ]    

def create_vectors_from_cymatics(
    path_gif:str,
    thumbnail_size_px:int=15,
    crop_left_px:int=225,
    crop_right_px:int=375,
    crop_bottom_px:int=75,
    crop_top_px:int=225,
    binary_threshold:int=128
) -> Generator[ndarray,None,None]:
    with Image.open(path_gif) as image:
        for i in range(image.n_frames):
            image.seek(i) 
            pixels = array(image.convert('L')) > binary_threshold
            note_pixels = pixels[crop_bottom_px:crop_top_px, crop_left_px:crop_right_px].astype(uint8)
            thumbnail_note_pixels = resize(note_pixels, (thumbnail_size_px, thumbnail_size_px), interpolation=INTER_AREA)
            yield thumbnail_note_pixels.flatten()            
            #imshow(note_pixels, 'gray')
            #show()
            #imshow(thumbnail_note_pixels, 'gray')
            #show()

music_vectors = dict(zip(
    notenames(),
    create_vectors_from_cymatics(path_gif='cymatics.gif')
))


from matplotlib.pyplot import imshow, show
spacetime = []
happy_birthday = [
    "G4","G4","A4","G4","C5","B4",
    "G4","G4","A4","G4","D5","C5",
    "G4","G4","E5","C5","B4","A4",
    "F5","F5","E5","C5","D5","C5"
]
for note in happy_birthday:
    spacetime.append(music_vectors[note])

imshow(spacetime)
show()
