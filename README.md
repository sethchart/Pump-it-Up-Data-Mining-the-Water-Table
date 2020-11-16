# Project Description
In this post, I will tell you about building my submission to the [Pump it Up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) competition hosted by [DrivenData](https://www.drivendata.org/). The competition problem uses data about water pumps in Tanzania collected by a partnership between [Taarifa](http://taarifa.org/) and the [United Republic of Tanzania Ministry of Water](https://www.maji.go.tz/) to predict if they are currently in need of repair. Specifically, the goal is to classify a water pump as 'functional', 'functional needs repair', or 'non functional' given the available data. By better understanding what factors predict issues with water pumps, we hope to improve access to potable water in Tanzania and reduce operational costs associated with maintaining water infrastructure. 

For the purposes of the competition, the model is assessed in terms of the classification rate, also commonly called [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision).
The classification rate is a number between zero and one with zero indicating that all prediction were wrong, one indicating all predictions were correct, and 0.7 indicating that, on average, seven out of ten predictions are correct.

My model produced an accuracy score of 0.8217. As of the time of writing, the best reported score for the competition is 0.8294 and my model ranks 490 out of 10,304 submissions.

# Main Takeaway
I was surprised by the performance that I was able to achieve without any feature engineering or manual data cleaning. This model is an exercise in applying off-the-shelf tools with minimal effort to great effect. 

My main model can be built by running the [FinalModel](code/FinalModel.ipynb) notebook.
