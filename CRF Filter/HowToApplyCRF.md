Here provides an explanation on how to apply the CRF filter on the top of Faster R-CNN when doing inference.


1. Obtain prediction results from the Faster R-CNN (there is no change in this step):

    ...
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(image_path)
    outputs = predictor(image)
    
    results_fasterRCNN = outputs["instances"]._fields
    
2. Send the results to CRF filter and apply the filter:

    Filter = RF_filter(results_fasterRCNN)  
    results_CRFFilter = test.filter_main()
    

the upadted predictions will be stored in results_CRFFilter.

Sample codes can be found in this folder
 
