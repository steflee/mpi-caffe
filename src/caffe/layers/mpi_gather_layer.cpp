#include <cfloat>
#include <vector>
#include <typeinfo>

#include "caffe/layers/mpi_gather_layer.hpp"

namespace caffe {



template <typename Dtype>
bool MPIGatherLayer<Dtype>::MPISyncFlag(bool flag){
	int temp = (int)flag;
	MPI_Bcast(&temp,1,MPI_INT, this->comm_root_, this->comm_);	
	return (bool)temp;
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top){

	

	int root=this->comm_root_;
	if(this->comm_rank_ == root){
		CHECK_EQ(this->comm_size_, top.size());
		for(int i = 0; i < top.size(); i++)
			top[i]->ReshapeLike(*bottom[0]);
	}
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	
	int root=this->comm_root_;
	if(this->comm_rank_ != root){
		for(int i = 0; i < top.size(); i++)
			top[i]->ReshapeLike(*bottom[0]);
	}
	
}


template <typename Dtype>
void MPIGatherLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
	
	int recv=this->comm_root_;
	Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	int count = bottom[0]->count();

	
	if(this->comm_rank_ == recv){ //If I am gatherer, collect all bottoms
		//Forward my bottom
		caffe_copy(count, bottom_data, top[this->comm_rank_]->mutable_cpu_data());
	
		//Collect other bottoms
		for(int i = 0; i < this->comm_size_; i++){
			if(i != recv){
				Dtype *top_data = top[i]->mutable_cpu_data();
				if(typeid(Dtype) == typeid(double))
					MPI_Recv(top_data, count, MPI_DOUBLE, i, 0, this->comm_,MPI_STATUS_IGNORE);
				else
					MPI_Recv(top_data, count, MPI_FLOAT, i, 0, this->comm_,MPI_STATUS_IGNORE);
			}		
		}
	}else{	//If I am not gatherer, send bottom
		
		if(typeid(Dtype) == typeid(double))
			MPI_Send(bottom_data, count, MPI_DOUBLE, recv, 0, this->comm_);
		else
			MPI_Send(bottom_data, count, MPI_FLOAT, recv, 0, this->comm_);

	}

}

template <typename Dtype>
void MPIGatherLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	int recv=this->comm_root_;
	Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
	int count = bottom[0]->count();

	if(this->comm_rank_ == recv){ //If I am gatherer, send gradients back
		//Forward my top grads back
		caffe_copy(count, top[this->comm_rank_]->mutable_cpu_diff(), bottom_diff);

		//Send out other top grads 
		for(int i = 0; i < this->comm_size_; i++){
			if(i != recv){
				Dtype *top_data = top[i]->mutable_cpu_diff();
				if(typeid(Dtype) == typeid(double))
					MPI_Send(top_data, count, MPI_DOUBLE, i, 0, this->comm_);
				else
					MPI_Send(top_data, count, MPI_FLOAT, i, 0, this->comm_);
			}		
		}
	}else{ //if i am not gatherer, recieve gradients
		if(typeid(Dtype) == typeid(double))
			MPI_Recv(bottom_diff, count, MPI_DOUBLE, recv, 0, this->comm_,MPI_STATUS_IGNORE);
		else
			MPI_Recv(bottom_diff, count, MPI_FLOAT, recv, 0, this->comm_,MPI_STATUS_IGNORE);

	}
}


INSTANTIATE_CLASS(MPIGatherLayer);
REGISTER_LAYER_CLASS(MPIGather);

}  // namespace caffe
