import snappy
import spherogram

pd_code_1 = [
    [2, 5, 3, 6],
    [4, 10, 5, 9],
    [6, 11, 7, 12],
    [8, 1, 9, 2],
    [10, 4, 11, 3],
    [12, 7, 1, 8]
]

pd_code_2 = [
    [1, 8, 2, 9],
    [3, 11, 4, 10],
    [5, 1, 6, 12],
    [7, 2, 8, 3],
    [9, 7, 10, 6],
    [11, 5, 12, 4]
]

link1 = spherogram.Link(pd_code_1)
link2 = spherogram.Link(pd_code_2)

# Convert to DT codes
dt_code_1 = link1.DT_code()
dt_code_2 = link2.DT_code()

print("DT code of Link 1:", dt_code_1)
print("DT code of Link 2:", dt_code_2)

