- All training results such as trained parameters file(`.pth`), log files(`.csv` ) will be stored as the following structure.
	
~~~
|—— 📁 logs 
	|—— 📁 ResNet34 (Used CNN Name)
		|—— 📁 cifar-100-balanced (Used Dataset Name)
			|—— 📁 ERM (Used Loss Function Name)
				|—— 📁 2020-08-01-03-30-26 (Start Time of Training)
					|—— model.pth (Trained Parameters File)
                    			|—— train.csv (Log for Training)
                   			|—— valid.csv (Log for Validation)
                   			|—— event files for tensorboard summary
	|—— 📁 ...
~~~
