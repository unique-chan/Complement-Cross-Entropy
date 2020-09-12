1. Add your own CNN model file (`*.py`) here.
2. You then have to import your model in `get_model.py` (line 3).
3. Finally, add your model in dictionary of `get_model()` in `get_model.py` (line 6).

~~~
def get_model(model_index):
    model_dic = {
        0: torch_models.alexnet,
        1: torch_models.vgg11,
        ...
        integer_key: your_model           # like this!
    }
~~~
