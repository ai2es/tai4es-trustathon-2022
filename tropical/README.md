In this notebook, we redirect you towards the tropical cyclone dataset hosted by Radiant MLHub. There are two jupyter notebooks which can be used to 
1) download, as well as extract the training and testing data as well as the labels, 
2) perform data exploration, train ML models, as well as save the best model and evaluate the results. 

Dependencies and Authentication
In order to access the dataset from radiant earth, you will need to register for an MLHub API key at https://mlhub.earth
Do not share your API key with anyone. Once you have your API key, you will need to create a default profile by setting up a .mlhub/profiles file in 
your home directory. Run the following commands to set up your profile

pip install radiant_mlhub
$ mlhub configure
API Key: Enter your API key here...
Wrote profile to /home/user/.mlhub/profiles

You will be able to run the following notebook once the above steps are completed successfully. 
It is faster to run the notebooks on colab and the process is much slower on local machine.

The notebooks can be accessed here:
https://github.com/radiantearth/mlhub-tutorials/tree/main/notebooks/NASA%20Tropical%20Storm%20Wind%20Speed%20Challenge 

Other useful links:
https://radiant-mlhub.readthedocs.io/en/latest/ 
