#include <iostream>
#include <vector>
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
    Matrix operator*(Matrix mat2)
    {
        if(col==mat2.row){
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
       
    }
    
     Matrix operator +(Matrix mat2)
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
    
    Matrix operator -(Matrix mat2)
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
    
    
    Matrix multiply(float x)
    {
        if(col==mat2.col && row==mat2.row){
            Matrix output;
            output.initialize(row,col);
            int r=row;
            int c=col;
            for (int i=0;i<r;i++){
                for (int j=0;j<c;j++){
                    output.values[output.ind(i,j)]=value_at(i,j)*x
                }
            }
            return output;
        }
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
    virtual Matrix Forward() = 0;
    virtual Matrix Backward() = 0;
    
    
};

class Linear:Layer
{
    public:
    
    Matrix weight;
    public:
    Linear(int prev_dimension,int curr_dimension)
    {
        
        weight.initialize(prev_dimension,curr_dimension,1);
        //Matrix bias(1,curr_dimension);
    }
    Matrix Forward(Matrix input)
    {
        Matrix out=input*weight;
        return out
    }
    Matrix Backward(Matrix input ,Matrix error)
    {
        
        
    }
};
class ANN
{
    public :
    vector<Linear> Model;
    vector<Matrix> inputs;//input at each layer 
};
int main() {
    // Write C++ code here
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
