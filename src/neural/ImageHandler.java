package neural;

import javax.imageio.ImageIO;
import javax.swing.*;


import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImageHandler {
    public static int[] ImageToIntArray(File file) throws IOException {
        BufferedImage img = ImageIO.read(file);
        int width = img.getWidth(), height = img.getHeight();
        int[] imgArr = new int[height*width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                imgArr[i*width+j] = img.getData().getSample(j, i, 0);
            }
        }
        return imgArr;
    }
    
    public static void scaleImage(String inputImagePath , String outputImagePath) throws IOException {
    	
    	File inputFile = new File(inputImagePath);
        BufferedImage scaledImage = ImageIO.read(inputFile);
    	
    	//BufferedImage scaledImage = new BufferedImage(40,
               // 40, inputImage.getType());
    	
    	for(int i = 0; i < scaledImage.getHeight(); ++i) {
            
            for(int j = 0; j < scaledImage.getWidth(); ++j) {
            
               Color c = new Color(scaledImage.getRGB(j, i));
               int red = (int)(c.getRed() * 0.299);
               int green = (int)(c.getGreen() * 0.587);
               int blue = (int)(c.getBlue() *0.114);
               Color newColor = new Color(red+green+blue,red+green+blue,red+green+blue);
               
               scaledImage.setRGB(j,i,newColor.getRGB());
            }
         }
    	
    	// extracts extension of output file
        String formatName = outputImagePath.substring(outputImagePath
                .lastIndexOf(".") + 1);
 
        // writes to output file
        ImageIO.write(scaledImage, formatName, new File(outputImagePath));
        
        //System.out.println("img: " + outputImagePath);
        
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src = Imgcodecs.imread(outputImagePath);

        //Creating an empty matrix to store the result
        Mat dst = new Mat();

        //Scaling the Image
        Imgproc.resize(src, dst, new Size(40, 40), 0, 0, 
           Imgproc.INTER_AREA);

        //Writing the image
        Imgcodecs.imwrite(outputImagePath, dst);
    	
    }

    public static void showImage(String filename) throws IOException {
        BufferedImage img = ImageIO.read(new File(filename));
        JFrame frame=new JFrame();
        ImageIcon icon=new ImageIcon(img);
        frame.setSize(img.getWidth()*5, img.getHeight()*5);
        JLabel lbl=new JLabel(icon);
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}