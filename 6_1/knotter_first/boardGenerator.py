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
                output_array.append([[i[3],i[0],i[1],i[2]]])
            else:
                output_array.append([[i[1],i[2],i[3],i[0]]])
        
        return output_array
        
# Example usage on the 6_2 knot:
if __name__ == "__main__":
    input_array = 	    	[[1,7,2,6],[3,10,4,11],[5,3,6,2],[7,1,8,12],[9,4,10,5],[11,9,12,8]]
    generator = BoardGenerator(input_array)
    output_array = generator.generate_board()
    print(("Input Array: 3_1", input_array))
    print(("Output Array:", output_array))
    