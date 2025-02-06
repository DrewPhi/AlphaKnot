# Define the BoardGenerator class
from __future__ import print_function
class BoardGenerator:
    def __init__(self, input_array):
        self.input_array = input_array
    def isPositive(self, crossing):
        #Given a crossing of the planar diagram code this outputs wether it is a positive or negative crossing.
        return crossing[1]>crossing[3]
    def generate_board(self):
        #Given a pd_code of a knot this generates the pd_code of the knot shadow.
        #Each crossing is represented by the two possible states that the neural network would then pick from.
        output_array = []
        for i in self.input_array:
            if self.isPositive(i):
                output_array.append([i,[i[3],i[0],i[1],i[2]]])
            else:
                output_array.append([i,[i[1],i[2],i[3],i[0]]])
        
        return output_array
    
# Example usage on the 6_1 knot:
if __name__ == "__main__":
    input_array = 	[[1,5,2,4],[3,1,4,6],[5,3,6,2]]
    generator = BoardGenerator(input_array)
    output_array = generator.generate_board()
    print(("Input Array: 3_1", input_array))
    print(("Output Array:", output_array))
    input_array = 	[[4,2,5,1],[8,6,1,5],[6,3,7,4],[2,7,3,8]]
    generator = BoardGenerator(input_array)
    output_array = generator.generate_board()
    print(("Input Array:4_1", input_array))
    print(("Output Array:", output_array))
    input_array = 	[[2,8,3,7],[4,10,5,9],[6,2,7,1],[8,4,9,3],[10,6,1,5]]
    generator = BoardGenerator(input_array)
    output_array = generator.generate_board()
    print(("Input Array:5_1", input_array))
    print(("Output Array:", output_array))
    input_array = 		[[1,7,2,6],[3,10,4,11],[5,3,6,2],[7,1,8,12],[9,4,10,5],[11,9,12,8]]
    generator = BoardGenerator(input_array)
    output_array = generator.generate_board()
    print(("Input Array:6_1", input_array))
    print(("Output Array:", output_array))
    input_array = 		[[2,10,3,9],[4,14,5,13],[6,12,7,11],[8,2,9,1],[10,8,11,7],[12,6,13,5],[14,4,1,3]]
    generator = BoardGenerator(input_array)
    output_array = generator.generate_board()
    print(("Input Array:7_2", input_array))
    print(("Output Array:", output_array))
    input_array = 		[[1,9,2,8],[3,7,4,6],[5,12,6,13],[7,3,8,2],[9,1,10,16],[11,15,12,14],[13,4,14,5],[15,11,16,10]]
    generator = BoardGenerator(input_array)
    output_array = generator.generate_board()
    print(("Input Array:8_1", input_array))
    print(("Output Array:", output_array))
