// Online C++ compiler to run C++ program online
#include <iostream>
#include <vector>
using namespace std;

class Matrix{
  public :    
    int col;
    int row;
    vector<float> values;
  public:
    Matrix(int r, int c)
    {
        values.resize(r*c);
        for (int i =0; i < r*c; i++){
              int b = rand() % 20 + 1;
              values.push_back(b);
              cout << values[i] << endl;
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
        int arr[2];
 
        arr[0] = row;
        arr[1] = col;
        return arr;
    }
    
    Matrix operator*(Matrix mat2)
    {
        if(col==mat2.row){
            Matrix output(row,mat2.col);
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
            Matrix output(row,col);
            int r=row;
            int c=col;
            for (int i=0;i<r;i++){
                for (int j=0;j<c;j++){
                    float result=0.0f;
                    output.values[output.ind(i,j)]=value_at(i,j)+mat2.value_at(i,j);
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

    

class Linear
{   public:
    
    Linear(int prev_dimension,int curr_dimension)
    {
        Matrix weight(prev_dimension,curr_dimension);
        Matrix bias(1,curr_dimension);
        
        
    }
};
class ANN
{
    public :
    vector<int> Model;
    vector<Matrix> inputs;//input at each layer 
};
int main() {
    // Write C++ code here
    Matrix test(5,6);
    return 0;
}
