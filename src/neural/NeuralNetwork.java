package neural;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

public class NeuralNetwork {
	
	int epochs;
	int numOfInput;
	int numOfheddin;
	int numOfOutput;
	float learningRate;
	
	double meanSquareError = 0 , error_sum = 0;
	
	double[] input , hidden , output , deltaHidden , deltaOutput;
	
	 int[][] trainingSetFeatures, testingSetFeatures;
     int[] trainingSetLabels, testingSetLabels;
     
     double[][] hiddenWeight , outWeight;
	
	NeuralNetwork() {
		
	}
	
	double[][] initWeights(int d1 , int d2) {
		
		double[][] w = new double[d1][d2];
		
		Random rand = new Random();
		
		for(int i = 0; i < d1; ++i) {
			
			for(int j = 0; j < d2; ++j) {
				
				double r = rand.nextDouble();
				
				w[i][j] = (r < 0.5) ? ((double)(0 + (255 - 0) * rand.nextDouble()) / (double)255) * -1
						: ((double)(0 + (255 - 0) * rand.nextDouble()) / (double)255);
			
			}
			
		}
		
		return w;
	}
	
	public void setNN(int e , int in , int hidd , int out , float r) {
		this.epochs = e;
		this.numOfInput = in;
		this.numOfheddin = hidd;
		this.numOfOutput = out;
		this.learningRate = r;
		
		input = new double[in];
		hidden = new double[hidd];
		output = new double[out];
		
		deltaHidden = new double[hidd];
		deltaOutput = new double[out];
		
		//initiate hidden weights
		hiddenWeight = new double[hidd][in];
		this.hiddenWeight = this.initWeights(hidd, in);
		
		//initiate output weights
		outWeight = new double[out][hidd];
		this.outWeight = this.initWeights(out, hidd);
		
	}
	
	public double sigmoid(double net) {
				
		double expo = Math.exp(-1 * net);
				
		return (1 / (1  + expo));
	}
	
	public void forwardPropagation() {
		
		double netHidden = 0 , netOut = 0;
		
		//calculating value for each hidden layer neuron j
		for(int j = 0; j < this.hidden.length; ++j) {
			
			//reset net
			netHidden = 0;
			
			//calculating net for each hidden neuron
			for(int i = 0; i < this.input.length; ++i) {
				
				netHidden += this.hiddenWeight[j][i] * this.input[i];				
			}
			
			//setting hidden neuron value with sigmoid activation function
			this.hidden[j] = this.sigmoid(netHidden);			
		}
		
		//calculating value for each output layer neuron k
		for(int k = 0; k < this.output.length; ++k) {
			
			//reset output net
			netOut = 0;
			
			//calculating net for each output neuron
			for(int j = 0; j < this.hidden.length; ++j) {
				
				netOut += this.outWeight[k][j] * this.hidden[j];
				
			}
			
			//setting output neuron value with sigmoid activation function
			this.output[k] = this.sigmoid(netOut); //<= 0.5? 0 : 1			
		}
		
	}
	
	public void backwordPropagation(int y) {
				
		//for each output neuron calculate deltaOutput(only one output neuron which is y)
		for(int k = 0; k < this.output.length; ++k) {
			this.deltaOutput[k] = (this.output[k] - y) * this.output[k] * (1 - this.output[k]);
		
		}
		
		
		double net = 0;
		//for each hidden neuron j caculate deltaHidden
		for(int j = 0; j < this.hidden.length; ++j) {
			
			net = 0;
			//calculating sum of deltaOutput multiply outWeight for all output neuron
			for(int k = 0; k < this.output.length; ++k)
				net += this.deltaOutput[k] * this.outWeight[k][j];
			
			this.deltaHidden[j] = net * this.hidden[j] * (1 - this.hidden[j]);
			
		}
		
		//updating outputWeight for each output neuron
		for(int k = 0; k < this.output.length; ++k) {
			
			for(int j = 0; j < this.hidden.length; ++j) {
				
				this.outWeight[k][j] = this.outWeight[k][j]
						- (this.learningRate * this.deltaOutput[k] * this.hidden[j]);
								
			}
			
		}
		
		//updating hiddenWeight for each hidden neuron
		for(int j = 0; j < this.hidden.length; ++j) {
			
			for(int i = 0; i < this.input.length; ++i) {
				
				this.hiddenWeight[j][i] = this.hiddenWeight[j][i]
						- (this.learningRate * this.deltaHidden[j] * this.input[i]);
				
			}
			
		}
		
	}
	
	public double getMeanSquareError() {
		
		return this.meanSquareError;
		
	}
	
	public void train(int[][] trainingFeatures , int[] trainingLabels) {
		
		this.trainingSetFeatures = new int[trainingFeatures.length][trainingFeatures[0].length];
		this.trainingSetLabels = new int[trainingLabels.length];
		
		this.trainingSetFeatures = trainingFeatures;
		this.trainingSetLabels = trainingLabels;
		
		
		//performing forward && backward propagation for number of epochs for each training example
		for(int e = 0; e < this.epochs; ++e) {
			
			error_sum = 0;
			
			for(int trainingExample = 0; trainingExample < this.trainingSetFeatures.length; ++trainingExample) {
				
				//copy training example to input featurs neures
				for(int x = 0; x < trainingSetFeatures[trainingExample].length; ++ x) {
					
					this.input[x] = (double)((double)this.trainingSetFeatures[trainingExample][x] / (double)255);
					
				}
					
				this.forwardPropagation();
				
				this.backwordPropagation(this.trainingSetLabels[trainingExample]);
				
				//calculating error(actual label - predicted output)
				double num = this.trainingSetLabels[trainingExample] - this.output[0];
				error_sum += num * num;
			}
			
			meanSquareError = error_sum / this.trainingSetFeatures.length;
			
		}
		
	}
	
	public int predict(int[] sampleFeature) {
		
		this.input = new double[sampleFeature.length];
		this.hidden = new double[100];
		this.output = new double[1];
		
		for(int i = 0; i < sampleFeature.length; ++i)
			this.input[i] = ((double)sampleFeature[i] / (double)255);
		
		this.forwardPropagation();
		
		return this.output[0] < 0.5 ? 0 : 1;
		
	}
	//testing model
	public int[] predict(int[][] testingFeatures) {
		
		int[] predictedLabels = new int[testingFeatures.length];
		
		testingSetFeatures = new int [testingFeatures.length][testingFeatures[0].length];
		
		testingSetFeatures = testingFeatures;
		
		this.input = new double[this.testingSetFeatures[0].length];
		
		for(int i = 0; i < this.testingSetFeatures.length; ++i) {
			
			for(int j = 0; j < this.testingSetFeatures[0].length; ++j)
				input[j] = ((double)this.testingSetFeatures[i][j] / (double)255);
				
			this.forwardPropagation();
			
			System.out.println("out: " +  this.output[0]);
			
			predictedLabels[i] = this.output[0] < 0.5 ? 0 : 1;
			
		}
		
		return predictedLabels;
	}
	
	public double calculateAccuracy(int[] predictedLabels, int[] testingSetLabels) {
		
		int cnt = 0;
		
		for(int i = 0; i < predictedLabels.length; ++i) {
			
			System.out.println("predict: " + predictedLabels[i] + " test: " + testingSetLabels[i]);
			
			if(predictedLabels[i] == testingSetLabels[i])
				++cnt;
		}
		
		return ((double)((double)cnt / (double)predictedLabels.length) * 100);
		
	}
	
	public String getWeights(double[][] w) {
		
		String s = "";
				
		for(int i = 0; i < w.length; ++i) {
			
			for(int j = 0; j < w[i].length; ++j) {
				
				if(j == w[i].length - 1)
					s += w[i][j] + "\n";
				else
					s += w[i][j] + " ";
				
			}
									
		}
		
		return s;
	}
	
	public void save(String fileName) {
		
		try (FileWriter f = new FileWriter(fileName, true);
				BufferedWriter b = new BufferedWriter(f);
				PrintWriter p = new PrintWriter(b);) {
						
			String s = this.getWeights(this.hiddenWeight) + this.getWeights(this.outWeight);
						
			p.print(s);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public void load(String fileName) {
		
		try {
	         BufferedReader in = new BufferedReader(new FileReader(fileName));
	         String str;
	         
	         int currentIHidden = 0 , currentIOut = 0;
	         
	         this.hiddenWeight = new double[100][1600];
	         this.outWeight = new double[1][100];
	         
	         while ((str = in.readLine()) != null) {
	        	 
	        	 String[] strArr = str.split(" ");
	        	 
	        	 if(currentIHidden < 100) {
	        		 
	        		 for(int j = 0; j < strArr.length; ++j) {
	        			 
	        			 this.hiddenWeight[currentIHidden][j] = Double.parseDouble(strArr[j]);
	        			 
	        		 }
	        		 
	        		 ++currentIHidden;
        		 
	        	 }else {
	        		 
	        		 for(int j = 0; j < strArr.length; ++j) {
	        			 
	        			 this.outWeight[currentIOut][j] = Double.parseDouble(strArr[j]);
	        			 
	        		 }
	        		 
	        		 ++currentIOut;
	        		 
	        	 }
	        	 
	         }
	      } catch (IOException e) {
	      }
		
	}
	

}
