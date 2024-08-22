
def create_vectors(
    n:int=4, 
    common_interval:int=720,
    note_names = ("C","C#","D","D#","E","F","F#","G","G#","A","A#","B"),
    relative_intervals = (720,675,640,600,576,540,512,480,450,432,405,384)
) -> dict[str,list[int]]:
    vectors = []
    vector_size = n*common_interval
    for i in relative_intervals:
        vector = [
            int(not(j%i)) for j in range(1,vector_size+1)
        ]
        vectors.append(vector)

    return dict(zip(note_names,vectors))

from matplotlib.pyplot import imshow, show 

music_vectors = create_vectors(
    common_interval=72,
    relative_intervals = (72,67,64,60,57,54,51,48,45,43,40,38)    
)
imshow(list(music_vectors.values()), 'gray')
show()

spacetime = []
happy_birthday = [
    "G","G","A","G","C","B",
    "G","G","A","G","D","C",
    "G","G","E","C","B","A",
    "F","F","E","C","D","C"
]
for note in happy_birthday:
    spacetime.append(music_vectors[note])

imshow(spacetime)
show()