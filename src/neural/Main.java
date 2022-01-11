package neural;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import javax.imageio.ImageIO;

class ImageData {
    int[] pixels;
    int label;
    
    ImageData(){
    	
    }

    ImageData(int imgHeight, int imgWidth) { pixels = new int[imgHeight*imgWidth]; }
    public void setPixels(int[] pixels) { this.pixels = pixels; }
    public void setLabel(int lbl) { label = lbl; }
}

public class Main {
    public static void main(String[] args) throws IOException {
        //Load Data
        File[] images= new File("Cats & Dogs Sample Dataset").listFiles();
        ImageData[] data = new ImageData[images.length];
        
        for (int i = 0; i < images.length; i++) {
        	
        	data[i] = new ImageData();
        	
            data[i].setPixels(ImageHandler.ImageToIntArray(images[i]));
            
            data[i].setLabel(images[i].getName().contains("cat")? 0 : 1);
        }

        //Shuffle
        List<ImageData> tempData = Arrays.asList(data);
        Collections.shuffle(tempData);
        tempData.toArray(data);
       // tempData.clear();

        //Split the data into training (75%) and testing (25%) sets
        int[][] trainingSetFeatures, testingSetFeatures;
        int[] trainingSetLabels, testingSetLabels;
        
        int splitPoint = (int) ((float)data.length * 0.75);
        
        trainingSetFeatures = new int[splitPoint][1600];
        testingSetFeatures = new int[data.length - splitPoint][1600];
        
        trainingSetLabels = new int[splitPoint];
        testingSetLabels = new int[data.length - splitPoint];
        
        for(int i = 0; i < data.length; ++i) {
        	
        	if(i < splitPoint) {
        		
        		for(int j = 0; j < data[i].pixels.length; ++j)
        			trainingSetFeatures[i][j] = data[i].pixels[j];
        		
        		trainingSetLabels[i] = data[i].label;
        		
        	}else {
        		
        		for(int j = 0; j < data[i].pixels.length; ++j)
        			testingSetFeatures[i - (splitPoint)][j] = data[i].pixels[j];
        		
        		testingSetLabels[i - (splitPoint)] = data[i].label;
        		
        	}
        	
        }
        
        /*

        //Create the NN
        NeuralNetwork nn = new NeuralNetwork();
        //Set the NN architecture
        
        nn.setNN(100, 1600, 100, 1, (float) 0.003);

        //Train the NN
        nn.train(trainingSetFeatures, trainingSetLabels);
        
        System.out.println("Training Mean Square Error: " + nn.getMeanSquareError());

        //Test the model
        int[] predictedLabels = nn.predict(testingSetFeatures);
        double accuracy = nn.calculateAccuracy(predictedLabels, testingSetLabels);
        
        System.out.println("Accuracy: " + accuracy);

        //Save the model (final weights)
        nn.save("model.txt");
        
        */

        //Load the model and use it on an image
        NeuralNetwork nn2 = new NeuralNetwork();
        nn2.load("model.txt");
        
        File f = new File("Cats & Dogs Sample Dataset");
        String fileDir = f.getAbsolutePath();
        
        //scale any image to 40x40 in grayscale mood
        //ImageHandler.scaleImage(inputImgPath , outputImgPath);
     
        //int[] sampleImgFeatures = ImageHandler.ImageToIntArray(new File("cat0001.jpg"));
        
        String img = fileDir + "\\cats_00001.jpg";
        int[] sampleImgFeatures = ImageHandler.ImageToIntArray(new File(img));
        int samplePrediction = nn2.predict(sampleImgFeatures);
        ImageHandler.showImage(img);
        
        //Print "Cat" or "Dog"
        if(samplePrediction == 1)
        	System.out.println("Predicted to be DOG!");
        else if(samplePrediction == 0)
        	System.out.println("Predicted to be CAT!");
    }
}