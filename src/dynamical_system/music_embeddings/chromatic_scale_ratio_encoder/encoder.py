
def create_vectors(
    n_octaves:int=2,
    n:int=4, 
    common_interval:int=720,
    note_names = ("C","C#","D","D#","E","F","F#","G","G#","A","A#","B"),
    relative_intervals = (720,675,640,600,576,540,512,480,450,432,405,384)
) -> dict[str,list[int]]:
    assert n > n_octaves
    assert len(note_names)==len(relative_intervals)
    vectors = {}
    vector_size = n*common_interval
    for octave in range(1,n_octaves+1):
        for i,name in zip(relative_intervals,note_names):
            interval = i // octave
            vector = [int(not(j%interval)) for j in range(1,vector_size+1)]
            vectors[f"{name}{octave}"] = vector
    return vectors




from matplotlib.pyplot import imshow, show, yticks
from numpy import array 

music_vectors = create_vectors(
    common_interval=144,
    relative_intervals = (144,135,128,120,115,108,102,96,90,86,81,77)    
)
imshow(list(music_vectors.values()), cmap='gray', aspect='auto')
yticks(array(list(range(len(music_vectors)))), list(music_vectors.keys()))
show()


imshow([music_vectors['A1'],music_vectors['A2']])
show()

spacetime = []
happy_birthday = [
    "G1","G1","A1","G1","C2","B1",
    "G1","G1","A1","G1","D2","C2",
    "G1","G1","E2","C2","B1","A1",
    "F1","F1","E2","C2","D2","C2"
]
for note in happy_birthday:
    spacetime.append(music_vectors[note])

imshow(spacetime)
show()