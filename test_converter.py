import torch.nn as nn
import torch
import torch.nn.functional as F

class BoundsChecker(nn.Module):
    def __init__(self, model, bounds):
        super(BoundsChecker, self).__init__()
        self.model = model
        self.bounds = bounds

        self.bounds_layer = nn.Linear(len(self.bounds[0]), len(self.bounds[0]) * 2 * len(bounds))
        self.junction_layer = nn.Linear(len(self.bounds[0]) * 2 * len(bounds), len(self.bounds))

        self.relu = nn.ReLU()

        nn.init.constant_(self.bounds_layer.weight, 0)
        nn.init.constant_(self.bounds_layer.bias, 0)

        nn.init.constant_(self.junction_layer.weight, 0)
        nn.init.constant_(self.junction_layer.bias, 0)

        for disjunction_idx, disjunction in enumerate(self.bounds):
            for bound_idx, bound in enumerate(disjunction):
                min_index = disjunction_idx * len(self.bounds[disjunction_idx]) * 2 + bound_idx * 2
                max_index = disjunction_idx * len(self.bounds[disjunction_idx]) * 2 + bound_idx * 2 + 1

                if isinstance(bound[0], float) and bound[0] > float('-inf'):
                    self.bounds_layer.weight.data[min_index][bound_idx] = -1.
                    self.bounds_layer.bias.data[min_index] = bound[0]
                elif isinstance(bound[0], str):
                    node_idx = int(bound[0].split('_')[1])
                    self.bounds_layer.weight.data[min_index][node_idx] = 1.
                    self.bounds_layer.weight.data[min_index][bound_idx] = -1.

                if isinstance(bound[1], float) and bound[1] < float('inf'):
                    self.bounds_layer.weight.data[max_index][bound_idx] = 1.
                    self.bounds_layer.bias.data[max_index] = -bound[1]
                elif isinstance(bound[1], str):
                    node_idx = int(bound[1].split('_')[1])
                    self.bounds_layer.weight.data[max_index][node_idx] = -1.
                    self.bounds_layer.weight.data[max_index][bound_idx] = 1.

        for disjunction_idx in range(len(self.bounds)):
            for bound_idx in range(len(self.bounds[0])):
                min_index = disjunction_idx * len(self.bounds[disjunction_idx]) * 2 + bound_idx * 2
                max_index = disjunction_idx * len(self.bounds[disjunction_idx]) * 2 + bound_idx * 2 + 1

                self.junction_layer.weight.data[disjunction_idx][min_index] = 1.
                self.junction_layer.weight.data[disjunction_idx][max_index] = 1.


    def forward(self, x):
        outputs = self.model.forward(x)

        outputs = self.bounds_layer(outputs)
        outputs = self.relu(outputs)

        outputs = self.junction_layer(outputs)

        output = torch.min(outputs)

        return output


class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()

    def forward(self, x):
        return torch.tensor([3., 7., 4.])


model = ExampleModel()
bounds = [
    [[float('-inf'), 'y_1'], [2., 7.], ['y_1', 9.]],  # Example with -inf and inf bounds
    [[float('-inf'), float('inf')], [1., 2.], [4., 6.]]
]
bounds_checker = BoundsChecker(model, bounds)
input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
result = bounds_checker(input_data)

print(result)
