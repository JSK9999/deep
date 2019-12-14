function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  // return sigmoid(x) * (1 - sigmoid(x));
  return y * (1 - y);
}


class NeuralNetwork {
  constructor(input_nodes, hidden_nodes,hidden2_nodes, output_nodes) {
     
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.hidden2_nodes=hidden2_nodes;
    this.output_nodes = output_nodes;
    
    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ho = new Matrix(this.hidden2_nodes, this.hidden_nodes);
    this.weights_h20= new Matrix(this.output_nodes,this.hidden2_nodes);

    this.weights_ih.randomize();
    this.weights_ho.randomize();
    this.weights_h20.randomize();
   
    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_ho= new Matrix(this.hidden2_nodes,1);
    this.bias_h20 = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_ho.randomize();
    this.bias_h20.randomize();
    
    this.learning_rate = 0.15;
  }

  feedforward(input_array) {

    //히든층의 출력을 생성하는데 인풋 배열로 부터 받은 값을 가중치와 곱하고 바이어스 값을 더한다 
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    //시그모이드 함수를 map함수를 이용하여 각 배열들을 순서대로 각각 곱해서 배열을 새로 만듬 
    hidden.map(sigmoid);
    

    let hidden2= Matrix.multiply(this.weights_ho, hidden);
    hidden2.add(this.bias_ho);
    hidden2.map(sigmoid);
    //아웃풋은 히든층의 출력 과 히든층을 곱함 바이어스 더하고 시그모이드함수를 맵함수 이용해서 배열의 값들을 갱신
    
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
   
    // 오차를 계산
    //오차는=t-y  subtract함수로 계산후 그 오차를 배열에 다시 리턴
    let output_errors = Matrix.subtract(targets, outputs);

    // let gradient = outputs * (1 - outputs);
    // 디시그모이드는 활성화함수 시그모이드의 도함수임 
    //그레디언트는 출력값을 시그모이드 도함수를 적용한것 (기울기) 에 학습률,에러 곱함
    let gradients = Matrix.map(outputs, dsigmoid);
    //거기에 오류값구함 
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);
    

    //델타2 계산 //h2로
    let hidden2_T = Matrix.transpose(hidden2);
    let weight_ho_deltas2 = Matrix.multiply(gradients,hidden2_T);
    this.weights_h20.add(weight_ho_deltas2);
    this.bias_h20.add(gradients);
    //은닉층 에러 계산
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
    //은닉층 에러 계산
    let who_t =Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, hidden_errors2);
    let hidden_gradient= Matrix.map(hidden, dsigmoid);

    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);




 /*
    //히든층전체 출력의 델타값은 그레디언트와 히든층트랜스포즈 행렬곱 왜냐하면 히등층에서 출력층으로 가는 델타Wij=학습률*e*디시그모이드*가중치행렬을구해야하니까 출력층에 들어오는 값을(은닉층에서 출력층) 행렬곱을 하려면 트랜스포즈해줘야함 계산
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(hidden2_gradient, hidden_T);

    // 델타의 가중치를 적용(더해줌)
    this.weights_ho.add(weight_ho_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_ho.add(hidden2_gradient);

    // 은닉층의 에러를 계산   가중치은닉층의출력을 트랜스포즈 E=W(t)2*e    시그모이드가 활성화함수일때 시그모이드의도함수(vi)=1 그래서 dj=ej 와 같다고 하셨음    -->은닉층의 E를 찾으려고하는것
    let who_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, hidden_errors2);

    // 히든층의 그레디언트를 계산 위에서 한거처럼 반복
    let hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);
    */
    // 입력층에서 출력층으로 가는 델타 계산
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);
    //히든층으로 들어가는 델타의 가중치를 적용
    this.weights_ih.add(weight_ih_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_h.add(hidden_gradient);           

    
    /////
  }
  
}