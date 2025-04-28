# Nvidia Neural Network ANN
An amateur academic project to practice Neural Network design and execution. The project is composed of 3 files: _NVDA.csv_, _nvidia_nn.pt_ and _MyNN.py_.

**NVDA.csv** is data from the [NVIDIA Stocks Data 2025](https://www.kaggle.com/datasets/meharshanali/nvidia-stocks-data-2025) dataset. It includes the date and stock price information recorded on a selection.

**MyNN.py** is my program which includes the neural network training architecture as well as the method of predicting the next stock prices.

The program then takes the saved neural network and uses it to predict the next day's Open, Low, High, and Close stock prices.

![image](https://github.com/user-attachments/assets/495b8676-7299-49f5-a61b-eafc74ff874c)

_Notice the predicted values do not follow expected logic (Low > High)._

**nvidia_nn.pt** is the saved neural network, trained over 200 epochs.

### Takeaways
- My ANN originally ran over 1999-2025, which logically saw an massive increase in Nvidia's stock prices. Thus it was not a good choice for my established architecture. The model would predict that values continue to increase.
- Netural Networks are not contrained to logic in the data, and only finds patterns. So it is expected that the relationship Low < High will not always stand, for example.
