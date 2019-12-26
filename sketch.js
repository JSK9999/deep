let nn;
let training_data = [{
  ////트레이닝 데이터 
  inputs: [6.3,    3.3,    6.0,    2.5],targets: [0,0,1]
},
{
  inputs: [5.8,    2.7,    5.1,    1.9],targets: [0,0,1]
},
{
  inputs: [7.1,    3.0,    5.9,    2.1],targets: [0,0,1]
},
{
  inputs: [6.3,    2.9,    5.6,    1.8],targets: [0,0,1]
},
{
  inputs: [6.5,    3.0,    5.8,    2.2],targets: [0,0,1]
},
{
  inputs: [7.6,    3.0,    6.6,    2.1],targets: [0,0,1]
},
{
  inputs: [4.9,    2.5,    4.5,    1.7],targets: [0,0,1]
},
{
  inputs: [7.3,    2.9,    6.3,    1.8],targets: [0,0,1]
},
{
  inputs: [6.7,    2.5,    5.8,    1.8],targets: [0,0,1]
},
{
  inputs: [7.2,    3.6,    6.1,    2.5],targets: [0,0,1]
},
{
  inputs: [6.5,    3.2,    5.1,    2.0],targets: [0,0,1]
},
{
  inputs: [6.4,    2.7,   5.3,    1.9],targets: [0,0,1]
},
{
  inputs: [6.8,    3.0,    5.5,    2.1],targets: [0,0,1]
},
{
  inputs: [5.7,    2.5,    5.0,    2.0],targets: [0,0,1]
},
{
  inputs: [5.8,    2.8,    5.1,    2.4],targets: [0,0,1]
},
{
  inputs: [6.4,    3.2,    5.3,    2.3],targets: [0,0,1]
},
{
  inputs: [6.5,    3.0,    5.5,    1.8],targets: [0,0,1]
},
{
  inputs: [7.7,    3.8,    6.7,    2.2],targets: [0,0,1]
},
{
  inputs: [7.7,    2.6,    6.9,    2.3],targets: [0,0,1]
},
{
  inputs: [6.0,    2.2,    5.0,    1.5],targets: [0,0,1]
},
{
  inputs: [6.9,    3.2,    5.7,    2.3],targets: [0,0,1]
},
{
  inputs: [5.6,    2.8,    4.9,    2.0],targets: [0,0,1]
},
{
  inputs: [7.7,    2.8,    6.7,    2.0],targets: [0,0,1]
},
{
  inputs: [6.3,    2.7,    4.9,    1.8],targets: [0,0,1]
},
{
  inputs: [6.7,    3.3,    5.7,    2.1],targets: [0,0,1]
},/////////////////////////////////////////////////////////////////////////
{
  inputs: [7.0,    3.2,    4.7,    1.4],targets: [0,1,0]
},
{
  inputs: [6.4,    3.2,    4.5,    1.5],targets: [0,1,0]
},
{
  inputs: [6.9,    3.1,    4.9,    1.5],targets: [0,1,0]
},
{
  inputs: [5.5,    2.3,    4.0,    1.3],targets: [0,1,0]
},
{
  inputs: [6.5,    2.8,    4.6,    1.5],targets: [0,1,0]
},
{
  inputs: [5.7,    2.8,    4.5,    1.3],targets: [0,1,0]
},
{
  inputs: [6.3,    3.3,    4.7,    1.6],targets: [0,1,0]
},
{
  inputs: [4.9,    2.4,    3.3,    1.0],targets: [0,1,0]
},
{
  inputs: [6.6,    2.9,    4.6,    1.3],targets: [0,1,0]
},
{
  inputs: [5.2,    2.7,    3.9,    1.4],targets: [0,1,0]
},
{
  inputs: [5.0,    2.0,    3.5,    1.0],targets: [0,1,0]
},
{
  inputs: [5.9,    3.0,    4.2,    1.5],targets: [0,1,0]
},
{
  inputs: [6.0,    2.2,    4.0,    1.0],targets: [0,1,0]
},
{
  inputs: [6.1,    2.9,   4.7,    1.4],targets: [0,1,0]
},
{
  inputs: [5.6,    2.9,    3.6,    1.3],targets: [0,1,0]
},
{
  inputs: [6.7,    3.1,    4.4,    1.4],targets: [0,1,0]
},
{
  inputs: [5.6,    3.0,    4.5,    1.5 ],targets: [0,1,0]
},
{
  inputs: [5.8,    2.7,    4.1,    1.0 ],targets: [0,1,0]
},
{
  inputs: [6.2,    2.2,    4.5,    1.5],targets: [0,1,0]
},
{
  inputs: [5.6,    2.5,    3.9,    1.1],targets: [0,1,0]
},
{
  inputs: [5.9,    3.2,    4.8,    1.8],targets: [0,1,0]
},
{
  inputs: [6.1,    2.8,    4.0,    1.3],targets: [0,1,0]
},
{
  inputs: [6.3,    2.5,    4.9,    1.5],targets: [0,1,0]
},
{
  inputs: [6.1,    2.8,    4.7,    1.2],targets: [0,1,0]
},
{
  inputs: [6.4,    2.9,    4.3,    1.3],targets: [0,1,0]
},
{
  inputs: [5.1,    3.5,    1.4,    0.2],targets: [1,0,0]
},///////////////////////////////////////////////////////////////
{
  inputs: [4.9,    3.0,   1.4,    0.2],targets: [1,0,0]
},
{
  inputs: [4.7,    3.2,    1.3,    0.2],targets: [1,0,0]
},
{
  inputs: [4.6,    3.1,    1.5,    0.2],targets: [1,0,0]
},
{
  inputs: [5.0,    3.6,    1.4,    0.2],targets: [1,0,0]
},
{
  inputs: [5.4,    3.9,    1.7,    0.4],targets: [1,0,0]
},
{
  inputs: [4.6,    3.4,    1.4,    0.3],targets: [1,0,0]
},
{
  inputs: [5.0,    3.4,    1.5,    0.2],targets: [1,0,0]
},
{
  inputs: [4.4,    2.9,    1.4,    0.2],targets: [1,0,0]
},
{
  inputs: [4.9,    3.1,    1.5,    0.1],targets: [1,0,0]
},
{
  inputs: [5.4,    3.7,    1.5,    0.2],targets: [1,0,0]
},
{
  inputs: [4.8,    3.4,    1.6,    0.2],targets: [1,0,0]
},
{
  inputs: [4.8,    3.0,    1.4,    0.1],targets: [1,0,0]
},
{
  inputs: [4.3,    3.0,    1.1,    0.1],targets: [1,0,0]
},
{
  inputs: [5.8,    4.0,   1.2,    0.2],targets: [1,0,0]
},
{
  inputs: [5.7,    4.4,    1.5,    0.4],targets: [1,0,0]
},
{
  inputs: [5.4,    3.9,    1.3,    0.4],targets: [1,0,0]
},
{
  inputs: [5.1,    3.5,    1.4,    0.3],targets: [1,0,0]
},
{
  inputs: [5.7,    3.8,    1.7,    0.3],targets: [1,0,0]
},
{
  inputs: [5.1,    3.8,    1.5,    0.3],targets: [1,0,0]
},
{
  inputs: [5.4,    3.4,    1.7,    0.2],targets: [1,0,0]
},
{
  inputs: [5.1,    3.7,    1.5,    0.4],targets: [1,0,0]
},
{
  inputs: [4.6,    3.6,    1.0,    0.2],targets: [1,0,0]
},
{
  inputs: [5.1,    3.3,    1.7,    0.5],targets: [1,0,0]
},
{
  inputs: [4.8,    3.4,    1.9,    0.2],targets: [1,0,0]
}];

////////////////테스트 데이터 넣음///////////////////
let testing_data = [{
  inputs: [7.2,  3.2,  6.0,  1.8],targets: [0,0,1]
},{
  inputs: [6.2,  2.8,  4.8,  1.8],targets: [0,0,1]
},{
  inputs: [6.1,    3.0,    4.9,    1.8],targets: [0,0,1]
},{
  inputs: [6.4,    2.8,    5.6,    2.1],targets: [0,0,1]
},{
  inputs: [7.2,    3.0,    5.8,    1.6],targets: [0,0,1]
},{
  inputs: [7.4,    2.8,    6.1,    1.9],targets: [0,0,1]
},{
  inputs: [7.9,    3.8,    6.4,    2.0],targets: [0,0,1]
},{
  inputs: [6.4,    2.8,    5.6,    2.2],targets: [0,0,1]
},{
  inputs: [6.3,    2.8,    5.1,    1.5],targets: [0,0,1]
},{
  inputs: [6.1,    2.6,    5.6,    1.4],targets: [0,0,1]
},{
  inputs: [7.7,    3.0,    6.1,    2.3],targets: [0,0,1]
},{
  inputs: [6.3,    3.4,    5.6,    2.4],targets: [0,0,1]
},{
  inputs: [6.4,    3.1,    5.5,    1.8],targets: [0,0,1]
},{
  inputs: [6.0,    3.0,    4.8,    1.8],targets: [0,0,1]
},{
  inputs: [6.9,    3.1,    5.4,    2.1],targets: [0,0,1]
},{
  inputs: [6.7,    3.1,    5.6,    2.4],targets: [0,0,1]
},{
  inputs: [6.9,    3.1,    5.1,    2.3],targets: [0,0,1]
},{
  inputs: [5.8,    2.7,    5.1,    1.9],targets: [0,0,1]
},{
  inputs: [6.8,    3.2,    5.9,    2.3],targets: [0,0,1]
},{
  inputs: [6.7,    3.3,    5.7,    2.5],targets: [0,0,1]
},{
  inputs: [6.7,    3.0,    5.2,    2.3],targets: [0,0,1]
},{
  inputs: [6.3,    2.5,    5.0,    1.9],targets: [0,0,1]
},{
  inputs: [6.5,    3.0,    5.2,    2.0],targets: [0,0,1]
},{
  inputs: [6.2,    3.4,    5.4,    2.3],targets: [0,0,1]
},{
  inputs: [5.9,    3.0,    5.1,    1.8],targets: [0,0,1]
}
/////////////////////////////////////////////////////////////////001
,{
  inputs: [6.6,    3.0,    4.4,    1.4],targets: [0,1,0]
},{
  inputs: [6.8,    2.8,    4.8,    1.4],targets: [0,1,0]
},{
  inputs: [6.7,    3.0,    5.0,    1.7],targets: [0,1,0]
},{
  inputs: [6.0,    2.9,    4.5,    1.5],targets: [0,1,0]
},{
  inputs: [5.7,    2.6,    3.5,    1.0],targets: [0,1,0]
},{
  inputs: [5.5,    2.4,    3.8,    1.1],targets: [0,1,0]
},{
  inputs: [5.5,    2.4,    3.7,    1.0],targets: [0,1,0]
},{
  inputs: [5.8,    2.7,    3.9,    1.2],targets: [0,1,0]
},{
  inputs: [6.0,    2.7,    5.1,    1.6],targets: [0,1,0]
},{
  inputs: [5.4,    3.0,    4.5,    1.5],targets: [0,1,0]
},{
  inputs: [6.0,    3.4,    4.5,    1.6],targets: [0,1,0]
},{
  inputs: [6.7,    3.1,    4.7,    1.5],targets: [0,1,0]
},{
  inputs: [6.3,    2.3,    4.4,    1.3],targets: [0,1,0]
},{
  inputs: [5.6,    3.0,    4.1,    1.3],targets: [0,1,0]
},{
  inputs: [5.5,    2.5,    4.0,    1.3],targets: [0,1,0]
},{
  inputs: [5.5,    2.6,    4.4,    1.2],targets: [0,1,0]
},{
  inputs: [6.1,    3.0,    4.6,    1.4],targets: [0,1,0]
},{
  inputs: [5.8,    2.6,    4.0,    1.2],targets: [0,1,0]
},{
  inputs: [5.0,    2.3,    3.3,    1.0],targets: [0,1,0]
},{
  inputs: [5.6,    2.7,    4.2,    1.3],targets: [0,1,0]
},{
  inputs: [5.7,    3.0,    4.2,    1.2],targets: [0,1,0]
},{
  inputs: [5.7,    2.9,    4.2,    1.3],targets: [0,1,0]
},{
  inputs: [6.2,    2.9,    4.3,    1.3],targets: [0,1,0]
},{
  inputs: [5.1,    2.5,    3.0,    1.1],targets: [0,1,0]
},{
  inputs: [5.7,    2.8,    4.1,    1.3],targets: [0,1,0]
}
///////////////////////////////////////////////////010
,{
  inputs: [5.0,    3.0,    1.6,    0.2],targets: [1,0,0]
},{
  inputs: [5.0,    3.4,    1.6,    0.4],targets: [1,0,0]
},{
  inputs: [5.2,    3.5,    1.5,    0.2],targets: [1,0,0]
},{
  inputs: [5.2,    3.4,    1.4,    0.2],targets: [1,0,0]
},{
  inputs: [4.7,    3.2,    1.6,    0.2],targets: [1,0,0]
},{
  inputs: [4.8,    3.1,    1.6,    0.2],targets: [1,0,0]
},{
  inputs: [5.4,    3.4,    1.5,    0.4],targets: [1,0,0]
},{
  inputs: [5.2,    4.1,    1.5,    0.1],targets: [1,0,0]
},{
  inputs: [5.5,    4.2,    1.4,    0.2],targets: [1,0,0]
},{
  inputs: [4.9,    3.1,    1.5,    0.2],targets: [1,0,0]
},{
  inputs: [5.0,    3.2,    1.2,    0.2],targets: [1,0,0]
},{
  inputs: [5.5,    3.5,    1.3,    0.2],targets: [1,0,0]
},{
  inputs: [4.9,    3.6,    1.4,    0.1],targets: [1,0,0]
},{
  inputs: [4.4,    3.0,    1.3,    0.2],targets: [1,0,0]
},{
  inputs: [5.1,    3.4,    1.5,    0.2],targets: [1,0,0]
},{
  inputs: [5.0,    3.5,    1.3,    0.3],targets: [1,0,0]
},{
  inputs: [4.5,    2.3,    1.3,    0.3],targets: [1,0,0]
},{
  inputs: [4.4,    3.2,    1.3,    0.2],targets: [1,0,0]
},{
  inputs: [5.0,    3.5,    1.6,    0.6],targets: [1,0,0]
},{
  inputs: [5.1,    3.8,    1.9,    0.4],targets: [1,0,0]
},{
  inputs: [4.8,    3.0,    1.4,    0.3],targets: [1,0,0]
},{
  inputs: [5.1,    3.8,    1.6,    0.2],targets: [1,0,0]
},{
  inputs: [4.6,    3.2,    1.4,    0.2],targets: [1,0,0]
},{
  inputs: [5.3,    3.7,    1.5,    0.2],targets: [1,0,0]
},{
  inputs: [5.0,    3.3,    1.4,    0.2],targets: [1,0,0]
}
///////////////////////////////////////////////100
]

//네트워크의 입력노드 은닉노드 출력노드의 개수 입력


function setup() {
    let nn = new NeuralNetwork(4,5,7,3);

    for ( let i =0; i< 5000; i++){
    for (data of training_data) {
        nn.train(data.inputs, data.targets);
    }
    for (data of testing_data) {
        nn.train(data.inputs, data.targets);
    }
   }
    
    
    console.log(nn.feedforward([7.2,    3.2,    6.0,    1.8]));
    console.log(nn.feedforward([6.2,    2.8,    4.8,    1.8]));
    console.log(nn.feedforward([6.1,    3.0,    4.9,    1.8]));
    console.log(nn.feedforward([6.4,    2.8,    5.6,    2.1]));
    console.log(nn.feedforward([7.2,    3.0,    5.8,    1.6]));
    console.log(nn.feedforward([7.4,    2.8,    6.1,    1.9]));
    console.log(nn.feedforward([7.9,    3.8,    6.4,    2.0]));
    console.log(nn.feedforward([6.4,    2.8,    5.6,    2.2]));
    console.log(nn.feedforward([6.3,    2.8,    5.1,    1.5]));
    console.log(nn.feedforward([6.1,    2.6,    5.6,    1.4]));
    console.log(nn.feedforward([7.7,    3.0,    6.1,    2.3]));
    console.log(nn.feedforward([6.3,    3.4,    5.6,    2.4]));
    console.log(nn.feedforward([6.4,    3.1,    5.5,    1.8]));
    console.log(nn.feedforward([6.0,    3.0,    4.8,    1.8]));
    console.log(nn.feedforward([6.9,    3.1,    5.4,    2.1]));
    console.log(nn.feedforward([6.7,    3.1,    5.6,    2.4]));
    console.log(nn.feedforward([6.9,    3.1,    5.1,    2.3]));
    console.log(nn.feedforward([5.8,    2.7,    5.1,    1.9]));
    console.log(nn.feedforward([6.8,    3.2,    5.9,    2.3]));
    console.log(nn.feedforward([6.7,    3.3,    5.7,    2.5]));
    console.log(nn.feedforward([6.7,    3.0,    5.2,    2.3]));
    console.log(nn.feedforward([6.3,    2.5,    5.0,    1.9]));
    console.log(nn.feedforward([6.5,    3.0,    5.2,    2.0]));
    console.log(nn.feedforward([6.2,    3.4,    5.4,    2.3]));
    console.log(nn.feedforward([5.9,    3.0,    5.1,    1.8]));
    /////////////////////////////////////////////////////////////////
    console.log('001');
    console.log(nn.feedforward([6.6,    3.0,    4.4,    1.4]));
    console.log(nn.feedforward([6.8,    2.8,    4.8,    1.4]));
    console.log(nn.feedforward([6.7,    3.0,    5.0,    1.7]));
    console.log(nn.feedforward([6.0,    2.9,    4.5,    1.5]));
    console.log(nn.feedforward([5.7,    2.6,    3.5,    1.0]));
    console.log(nn.feedforward([5.5,    2.4,    3.8,    1.1]));
    console.log(nn.feedforward([5.5,    2.4,    3.7,    1.0]));
    console.log(nn.feedforward([5.8,    2.7,    3.9,    1.2]));
    console.log(nn.feedforward([6.0,    2.7,    5.1,    1.6]));
    console.log(nn.feedforward([5.4,    3.0,    4.5,    1.5]));
    console.log(nn.feedforward([6.0,    3.4,    4.5,    1.6]));
    console.log(nn.feedforward([6.7,    3.1,    4.7,    1.5]));
    console.log(nn.feedforward([6.3,    2.3,    4.4,    1.3]));
    console.log(nn.feedforward([5.6,    3.0,    4.1,    1.3]));
    console.log(nn.feedforward([5.5,    2.5,    4.0,    1.3]));
    console.log(nn.feedforward([5.5,    2.6,    4.4,    1.2]));
    console.log(nn.feedforward([6.1,    3.0,    4.6,    1.4]));
    console.log(nn.feedforward([5.8,    2.6,    4.0,    1.2]));
    console.log(nn.feedforward([5.0,    2.3,    3.3,    1.0]));
    console.log(nn.feedforward([5.6,    2.7,    4.2,    1.3]));
    console.log(nn.feedforward([5.7,    3.0,    4.2,    1.2]));
    console.log(nn.feedforward([5.7,    2.9,    4.2,    1.3]));
    console.log(nn.feedforward([6.2,    2.9,    4.3,    1.3]));
    console.log(nn.feedforward([5.1,    2.5,    3.0,    1.1]));
    console.log(nn.feedforward([5.7,    2.8,    4.1,    1.3]));
    //////////////////////////////////////////////////////////////
    console.log('010');
    console.log(nn.feedforward([5.0,    3.0,    1.6,    0.2]));
    console.log(nn.feedforward([5.0,    3.4,    1.6,    0.4]));
    console.log(nn.feedforward([5.2,    3.5,    1.5,    0.2]));
    console.log(nn.feedforward([5.2,    3.4,    1.4,    0.2]));
    console.log(nn.feedforward([4.7,    3.2,    1.6,    0.2]));
    console.log(nn.feedforward([4.8,    3.1,    1.6,    0.2]));
    console.log(nn.feedforward([5.4,    3.4,    1.5,    0.4]));
    console.log(nn.feedforward([5.2,    4.1,    1.5,    0.1]));
    console.log(nn.feedforward([5.5,    4.2,    1.4,    0.2]));
    console.log(nn.feedforward([4.9,    3.1,    1.5,    0.2]));
    console.log(nn.feedforward([5.0,    3.2,    1.2,    0.2]));
    console.log(nn.feedforward([5.5,    3.5,    1.3,    0.2]));
    console.log(nn.feedforward([4.9,    3.6,    1.4,    0.1]));
    console.log(nn.feedforward([4.4,    3.0,    1.3,    0.2]));
    console.log(nn.feedforward([5.1,    3.4,    1.5,    0.2]));
    console.log(nn.feedforward([5.0,    3.5,    1.3,    0.3]));
    console.log(nn.feedforward([4.5,    2.3,    1.3,    0.3]));
    console.log(nn.feedforward([4.4,    3.2,    1.3,    0.2]));
    console.log(nn.feedforward([5.0,    3.5,    1.6,    0.6]));
    console.log(nn.feedforward([5.1,    3.8,    1.9,    0.4]));
    console.log(nn.feedforward([4.8,    3.0,    1.4,    0.3]));
    console.log(nn.feedforward([5.1,    3.8,    1.6,    0.2]));
    console.log(nn.feedforward([4.6,    3.2,    1.4,    0.2]));
    console.log(nn.feedforward([5.3,    3.7,    1.5,    0.2]));
    console.log(nn.feedforward([5.0,    3.3,    1.4,    0.2]));
    console.log('100');

}

