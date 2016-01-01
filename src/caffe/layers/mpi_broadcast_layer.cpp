#include <cfloat>
#include <vector>
#include <typeinfo>

#include "caffe/layers/mpi_broadcast_layer.hpp"

namespace caffe {




template <typename Dtype>
bool MPIBroadcastLayer<Dtype>::MPISyncFlag(bool flag){	
	/*int temp = (int)needBack;
	MPI_Bcast(&temp,1,MPI_INT, this->layer_param().mpi_param().root(), MPI_COMM_WORLD);	
	return (bool)temp;*/

	LOG(INFO) << this->comm_rank_ << " - " << flag;
	int temp = (int) flag;
	int buffer[this->comm_size_];
	MPI_Allgather(&temp, 1, MPI_INT, buffer, 1, MPI_INT, this->comm_);

	for(int i = 0; i < this->comm_size_; i++)
		if(buffer[i]==1) 
			return true;

	return false;
}

template <typename Dtype>
void MPIBroadcastLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top){

	int dims[4], src=this->comm_root_;
	if(this->comm_rank_ == src){
		dims[0] = bottom[0]->num();
		dims[1] = bottom[0]->channels();
		dims[2] = bottom[0]->height();
		dims[3] = bottom[0]->width();
	}
	MPI_Bcast(dims,4,MPI_INT, src, this->comm_);	
	top[0]->Reshape(dims[0], dims[1], dims[2], dims[3]);
}

template <typename Dtype>
void MPIBroadcastLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	
	int dims[4], src=this->comm_root_;
	if(this->comm_rank_ == src){
		dims[0] = bottom[0]->num();
		dims[1] = bottom[0]->channels();
		dims[2] = bottom[0]->height();
		dims[3] = bottom[0]->width();
	}
	MPI_Bcast(dims,4,MPI_INT, src, this->comm_);	
	top[0]->Reshape(dims[0], dims[1], dims[2], dims[3]);
}


template <typename Dtype>
void MPIBroadcastLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
	int src=this->comm_root_;

	if(this->comm_rank_ == src){
		Dtype* bottom_data = bottom[0]->mutable_cpu_data();
		int count = bottom[0]->count();

		if(typeid(Dtype) == typeid(double))
			MPI_Bcast(bottom_data,count,MPI_DOUBLE, src, this->comm_);
		else
			MPI_Bcast(bottom_data,count,MPI_FLOAT, src, this->comm_);

		caffe_copy(count, bottom_data, top[0]->mutable_cpu_data());

	}else{
		Dtype* top_data = top[0]->mutable_cpu_data();
		int count = top[0]->count();
		if(typeid(Dtype) == typeid(double))
			MPI_Bcast(top_data,count,MPI_DOUBLE, src, this->comm_);
		else
			MPI_Bcast(top_data,count,MPI_FLOAT, src, this->comm_);	
	}	
}

template <typename Dtype>
void MPIBroadcastLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	int src=this->comm_root_;

	if(this->comm_rank_ == src){
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype* top_diff = top[0]->mutable_cpu_diff();
		int count = bottom[0]->count();

		caffe_copy(count, top_diff, bottom_diff);
		for(int i = 0; i < this->comm_size_; i++){
			if(i != src){
				if(typeid(Dtype) == typeid(double))
					MPI_Recv(top_diff, count, MPI_DOUBLE, i, 0, this->comm_,MPI_STATUS_IGNORE);
				else
					MPI_Recv(top_diff, count, MPI_FLOAT, i, 0, this->comm_,MPI_STATUS_IGNORE);

				caffe_add(count, bottom_diff, top_diff, bottom_diff);
			}
		}
	}else{
		Dtype* top_diff = top[0]->mutable_cpu_diff();
		int count = top[0]->count();
		
		if(typeid(Dtype) == typeid(double))
			MPI_Send(top_diff, count, MPI_DOUBLE, src, 0, this->comm_);
		else
			MPI_Send(top_diff, count, MPI_FLOAT, src, 0, this->comm_);
	}
}

INSTANTIATE_CLASS(MPIBroadcastLayer);
REGISTER_LAYER_CLASS(MPIBroadcast);

}
