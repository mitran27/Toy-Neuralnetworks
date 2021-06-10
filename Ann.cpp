#include <iostream>
#include <vector>
# include <assert.h>

using namespace std;

class Matrix{
  public :    
    int col;
    int row;
    vector<float> values;
  public:
    void initialize(int r, int c,int random=0)
    {   row=r;
        col=c;
        srand( (unsigned)time( NULL ) );
        for (int i =0; i < r*c; i++){
              float r=(float)rand()/RAND_MAX;    
              if(random)
              values.push_back(r-0.5);
              else
              values.push_back(0);
              
              //cout << (float)values[i]<<r << endl;
    }
   
    }
    
    int ind(int r, int c) // get indice of vector if row, adn col is given
    {
        return r*col+c;
    }
    float value_at (int r,int c)
    {
        return values[r*col+c];
    }
    
    int*  shape()
    { 
        static int arr[2];
 
        arr[0] = row;
        arr[1] = col;
        return arr;
    }
    void print()
    {
        for (int i=0;i<row;i++){
                for (int j=0;j<col;j++){
                    cout<<value_at(i,j)<<endl;
                }
            }
    }
    
   
     Matrix operator +(Matrix& mat2)
    {
        if(col==mat2.col && row==mat2.row){
            Matrix output;
            output.initialize(row,col);
            int r=row;
            int c=col;
            for (int i=0;i<r;i++){
                for (int j=0;j<c;j++){
                    output.values[output.ind(i,j)]=value_at(i,j)+mat2.value_at(i,j);
                }
            }
            return output;
        }
    }
    
    Matrix operator -(Matrix& mat2)
    {
        if(col==mat2.col && row==mat2.row){
            Matrix output;
            output.initialize(row,col);
            int r=row;
            int c=col;
            for (int i=0;i<r;i++){
                for (int j=0;j<c;j++){
                    output.values[output.ind(i,j)]=value_at(i,j)-mat2.value_at(i,j);
                }
            }
            return output;
        }
    }
    
    Matrix Applyfunction(float (&func)(float))
    {
        
            Matrix output;
            output.initialize(row,col);
            int r=row;
            int c=col;
            for (int i=0;i<r;i++){
                for (int j=0;j<c;j++){
                    output.values[output.ind(i,j)]=
                    func(value_at(i,j));
                }
            }
            return output;
        
    }
    
    
    Matrix multiply(float x)
    {
            Matrix output;
            output.initialize(row,col);
            int r=row;
            int c=col;
            for (int i=0;i<r;i++){
                for (int j=0;j<c;j++){
                    output.values[output.ind(i,j)]=value_at(i,j)*x;
                }
            }
            return output;
        }
    
    
    
    
    
     Matrix operator*(Matrix& mat2)
    {
            assert(col==mat2.row);
            Matrix output;
            output.initialize(row,mat2.col);
            int r=row;
            int c=mat2.col;
            for (int i=0;i<r;i++){
                for (int j=0;j<c;j++){
                    float result=0.0f;
                    for (int k=0;k<c;k++){
                        result+=value_at(i,k)*mat2.value_at(k,j);
                    }
                    output.values[output.ind(i,j)]=result;
                }
            }
            
        
        return output;
       
    }
    
    
    
    
    
};
Matrix Transpose(Matrix mat)
{
    for (int i=0;i<mat.row;i++){
        for (int j=i+1;j<mat.col;j++){
            
            float temp=mat.value_at(i,j);
            mat.values[mat.ind(i,j)]=mat.value_at(j,i);
            mat.values[mat.ind(j,i)]=temp;
        }
        
        
    }
    return  mat;
}

    
class Layer
{
    public:
    virtual Matrix Forward(Matrix m) {
        return m;
    }
    virtual Matrix Backward(Matrix m) {
        return m;
    }
    virtual void summary() {
        cout<<"layer";
    }
    
    
};

class Linear:public Layer
{
    public:
    Matrix weight;
    Linear(int prev_dimension,int curr_dimension)
    {
        
        weight.initialize(prev_dimension,curr_dimension,1);
        //Matrix bias(1,curr_dimension);
    }
    Matrix Forward(Matrix input)
    {
        Matrix out=input*weight;
        
        return out;
    }
    Matrix Backward(Matrix input ,Matrix error,float lr)
    {
        
        Matrix tran_wt=Transpose(weight);
        Matrix input_error=error*tran_wt;
        
        Matrix tran_ip=Transpose(input);
        Matrix weight_error=tran_ip*error;
        
        Matrix err=weight_error.multiply(lr);
        weight=weight-err;
        
        return input_error;
        
        
    }
    void summary()
    {
        int *x;
        x=weight.shape();
        cout<<"linear  in-dim  : "<<*x<<" out-dim : "<<*(x+1);
    }
};
class ANN
{
    public :
    vector<Layer*> Model;
    vector<Matrix> inputs;//input at each layer 
    void add(Layer  *lay){
        Model.push_back(lay);
    }
    Matrix cost(Matrix &prediction ,Matrix &Target){
        Matrix err=Target-prediction
        return err
    }
    
    void epoch(Matrix Input,Matrix Output,float lr)
    {   Matrix ip=Input
        for (int i=0;i<Model.size();i++){
            inputs.push_back(ip)
            Matrix op=Model[i]->forward(ip)
            ip=op
        }
        Matrix err=cost(ip,Output)
        for (int i=Model.size()-1;i<=0;i--){
            Matrix ip_lay=inputs[i]
            err=Model[i]->backward(ip,err,lr)
        }
        
    }
    
    void summary(){
        
        for (int i=0;i<Model.size();i++){
            Model[i]->summary();
        }
    }
    
};
int main() {
    // Write C++ code here
    ANN model;
    Linear l1(5,6);
    model.add(&l1);
    model.summary();

    
    
    
    /*
    Matrix test1(3,3,1);
    cout<<"\n\n";
    Matrix test2(3,3,1);
    cout<<"\n\n";
    Matrix out=test1*test2;
    int *x;
    x=out.shape();
    cout<<*x<<*(x+1);
    //cout<<*(p+1);*/
    
    
    return 0;
}
