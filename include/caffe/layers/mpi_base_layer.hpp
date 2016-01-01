#ifndef CAFFE_MPI_BASE_LAYER_HPP_
#define CAFFE_MPI_BASE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class MPIBaseLayer : public Layer<Dtype> {
 public:
  explicit MPIBaseLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {

      //Set up MPI variables
      this->comm_ = (MPI_Comm)param.mpi_param().comm_id();
      this->group_ = (MPI_Group)param.mpi_param().group_id();
      MPI_Comm_rank(MPI_COMM_WORLD, &this->world_rank_);
      MPI_Comm_size(MPI_COMM_WORLD, &this->world_size_);
      MPI_Comm_rank(this->comm_, &this->comm_rank_);
      MPI_Comm_size(this->comm_, &this->comm_size_);


      //Verify root is in local communicator
      MPI_Group world_group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);

      int old_src = param.mpi_param().root();
      MPI_Group_translate_ranks(world_group, 1, &old_src, this->group_, &this->comm_root_);
      CHECK(MPI_UNDEFINED != this->comm_root_) << "MPI Root not listed included in layer group.";
    
	}
 protected:
  MPI_Comm comm_;
  MPI_Group group_;
  int comm_rank_;
  int comm_size_;
	int world_size_;
	int world_rank_;
  int comm_root_;
};

}  // namespace caffe

#endif
