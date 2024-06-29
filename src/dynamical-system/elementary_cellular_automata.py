

# eca_110 = CellularAutomata(
#     cell_states=2, //default=2
#     lattice_width=1000, //default=100
#     time_steps=1000, //default=100
#     transition_rule=110, //default=110
#     transition_table=dict(), //default = None
#     neighbourhood_radius=1 //default=1
#     initial_configuration=[11000101010111...], //default=random
#     initial_state=330222111 //default=random same as above
# )
# eca.array() //ndarray
# eca_110[20:] //ndarray
# eca_110[10] //array([10101010111...])
# print(eca_110) //string with black and white emojis
# eca.save('bla.txt')

#scipy convolve