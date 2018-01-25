# ESE650-Color-Segmentation

To segment images, place the images in the "Test_Set" folder and run the file segment.py. This by default uses the stored GMM model in the "lookupTable". If the saved model is not present (or you want to train it again), call the function train() from segment.py. This will assume that the data files are present in the DATA_FOLDER specified in segment.py (don't run without ensuring that first!). For more details about the problem and approach, see the file ESE650_Project1.pdf and Report.pdf.
