function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  // return sigmoid(x) * (1 - sigmoid(x));
  return y * (1 - y);
}


class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, hidden2_nodes, output_nodes) {
     
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.hidden2_nodes = hidden2_nodes;
    this.output_nodes = output_nodes;
    
    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ho = new Matrix(this.hidden2_nodes, this.hidden_nodes);
    this.weights_h20= new Matrix(this.output_nodes,this.hidden2_nodes);
    //////////////// /////////////////////////////////////////////
    this.weights_ih.randomize();
    this.weights_ho.randomize();
    this.weights_h20.randomize();
   
    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_ho= new Matrix(this.hidden2_nodes,1);
    this.bias_h20 = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_ho.randomize();
    this.bias_h20.randomize();
    
    this.learning_rate = 0.03;
  }

  feedforward(input_array) {

    //히든층의 출력을 생성
    let inputs = Matrix.fromArray(input_array);
    
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    //시그모이드 함수 
    hidden.map(sigmoid);
    

    let hidden2= Matrix.multiply(this.weights_ho, hidden);
    hidden2.add(this.bias_ho);
    hidden2.map(sigmoid);
    //아웃풋은 히든층의 출력 과 히든층을 곱
    
    let output = Matrix.multiply(this.weights_h20, hidden2);
    output.add(this.bias_h20);
    output.map(sigmoid);
    

   
    // toarray함수 이용해서 새배열에 입력을 하고  리턴 --> Sending back to the caller
    return output.toArray();
  }
  
  train(input_array, target_array) {
    
    let inputs = Matrix.fromArray(input_array); 

    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(sigmoid);

    let hidden2= Matrix.multiply(this.weights_ho,hidden);
    hidden2.add(this.bias_ho);
    hidden2.map(sigmoid);

    let outputs = Matrix.multiply(this.weights_h20, hidden2);
    outputs.add(this.bias_h20);
    outputs.map(sigmoid);

    let targets = Matrix.fromArray(target_array);


   
    let output_errors = Matrix.subtract(targets, outputs);


    let gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);
    




    //델타2 계산 //h2로
    let hidden2_T = Matrix.transpose(hidden2);
    let weight_ho_deltas2 = Matrix.multiply(gradients,hidden2_T);


    this.weights_h20.add(weight_ho_deltas2);
    this.bias_h20.add(gradients);



    //은닉2층 에러 계산
    let who_t2 =Matrix.transpose(this.weights_h20);
    let hidden_errors2 = Matrix.multiply(who_t2, output_errors);


    let hidden2_gradient= Matrix.map(hidden2, dsigmoid);
    hidden2_gradient.multiply(hidden_errors2);
    hidden2_gradient.multiply(this.learning_rate);



    //델타1
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(hidden2_gradient,hidden_T);


    this.weights_ho.add(weight_ho_deltas);
    this.bias_ho.add(hidden2_gradient);
    


    //은닉1층 에러 계산
    let who_t =Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t,hidden_errors2);
   
    let hidden_gradient= Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);


    //인풋쪽 델타
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas=Matrix.multiply(hidden_gradient,inputs_T);

    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hidden_gradient);


  }
  
}