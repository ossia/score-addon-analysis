#include "VariableMarkovOracleMFCCFactory.hpp"

#include <Media/Commands/ChangeAudioFile.hpp>
#include <Media/Sound/SoundModel.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>
#include <score/document/DocumentContext.hpp>

#include <core/document/Document.hpp>

#include <ossia/audio/drwav_handle.hpp>

#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QProgressDialog>

#include <cmath>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <AudioFile.h>
#include <Gist.h>

namespace VariableMarkovOracleMFCC
{
QString GainDoubler::title() const noexcept
{
  return QObject::tr("Variable Markov Oracle - MFCC");
}

UuidKey<Process::ProcessModel> GainDoubler::target() const noexcept
{
  return Metadata<ConcreteKey_k, Media::Sound::ProcessModel>::get();
}
class LibrarySubsystemVMO
{
public:
  std::vector<std::vector<double>>
  GenerateAudioFrames(std::string input_filename, int hop_size)
  {
    AudioFile<double> audioFile;
    audioFile.load(std::move(input_filename));
    int numSamples = audioFile.getNumSamplesPerChannel();
    int numFrames = numSamples / hop_size;
    std::vector<double> sample;
    sample.reserve(hop_size);
    std::vector<std::vector<double>> buffer_matrix;
    buffer_matrix.reserve(numFrames);
    int k = 0;
    for (int j = 0; j < numFrames; j++)
    {
      sample = {};
      int current_len = k + hop_size;
      while (k < current_len)
      {
        sample.push_back(audioFile.samples[0][k]);
        k++;
      }
      buffer_matrix.push_back(sample);
    }
    return buffer_matrix;
  }
};

/**
 * The FacadeVMO class provides a simple interface to the complex logic of one or
 * several subsystems. The FacadeVMO delegates the client requests to the
 * appropriate objects within the subsystem. The FacadeVMO is also responsible for
 * managing their lifecycle. All of this shields the client from the undesired
 * complexity of the subsystem.
 */
class FacadeVMO
{
protected:
  LibrarySubsystemVMO* librarySubsystem_;
  /**
   * Depending on your application's needs, you can provide the FacadeVMO with
   * existing subsystem objects or force the FacadeVMO to create them on its own.
   */
public:
  /**
   * In this case we will delegate the memory ownership to FacadeVMO Class
   */
  FacadeVMO(LibrarySubsystemVMO* librarySubsystem = nullptr)
  {
    this->librarySubsystem_ = librarySubsystem ?: new LibrarySubsystemVMO;
  }
  ~FacadeVMO() { delete librarySubsystem_; }
  /**
   * The FacadeVMO's methods are convenient shortcuts to the sophisticated
   * functionality of the subsystems. However, clients get only to a fraction of
   * a subsystem's capabilities.
   */
  std::vector<std::vector<double>>
  OperationGenerateFrames(std::string input_filename, int hop_size)
  {
    return this->librarySubsystem_->GenerateAudioFrames(
        std::move(input_filename), hop_size);
  }
};

/**
 * The client code works with complex subsystems through a simple interface
 * provided by the FacadeVMO. When a facade manages the lifecycle of the subsystem,
 * the client might not even know about the existence of the subsystem. This
 * approach lets you keep the complexity under control.
 */

std::vector<std::vector<double>> ClientCodeFramesVMO(
    FacadeVMO* facade,
    std::string input_filename,
    int hop_size)
{
  return facade->OperationGenerateFrames(std::move(input_filename), hop_size);
}
/**
 * The client code may have some of the subsystem's objects already created. In
 * this case, it might be worthwhile to initialize the FacadeVMO with these objects
 * instead of letting the FacadeVMO create new instances.
 */

/**
 * The StrategyVMO interface declares operations common to all supported versions
 * of some algorithm.
 *
 * The ContextVMO uses this interface to call the algorithm defined by Concrete
 * Strategies.
 */
class StrategyVMO
{
public:
  virtual ~StrategyVMO() { }
  virtual double
  DoVectorDistance(double* first, double* last, double* first2) const = 0;
};

/**
 * The ContextVMO defines the interface of interest to clients.
 */

class ContextVMO
{
  /**
   * @var StrategyVMO The ContextVMO maintains a reference to one of the StrategyVMO
   * objects. The ContextVMO does not know the concrete class of a strategy. It
   * should work with all strategies via the StrategyVMO interface.
   */
private:
  StrategyVMO* strategy_;
  /**
   * Usually, the ContextVMO accepts a strategy through the constructor, but also
   * provides a setter to change it at runtime.
   */
public:
  ContextVMO(StrategyVMO* strategy = nullptr)
      : strategy_(strategy)
  {
  }
  ~ContextVMO() { delete this->strategy_; }
  /**
   * Usually, the ContextVMO allows replacing a StrategyVMO object at runtime.
   */
  void set_strategy(StrategyVMO* strategy)
  {
    delete this->strategy_;
    this->strategy_ = strategy;
  }
  /**
   * The ContextVMO delegates some work to the StrategyVMO object instead of
   * implementing +multiple versions of the algorithm on its own.
   */
  double DoDistanceLogic(double* first, double* last, double* first2) const
  {
    double result = this->strategy_->DoVectorDistance(first, last, first2);
    return result;
  }
};

/**
 * Concrete Strategies implement the algorithm while following the base StrategyVMO
 * interface. The interface makes them interchangeable in the ContextVMO.
 */
class StrategyVMOEuclidean : public StrategyVMO
{
public:
  double
  DoVectorDistance(double* first, double* last, double* first2) const override
  {
    double ret = 0.0;
    while (first != last)
    {
      double dist = (*first++) - (*first2++);
      ret += dist * dist;
    }
    return ret > 0.0 ? sqrt(ret) : 0.0;
  }
};
class ConcreteStrategyVMOB : public StrategyVMO
{
  double
  DoVectorDistance(double* first, double* last, double* first2) const override
  {
    double ret = 0.0;
    while (first != last)
    {
      double dist = (*first++) - (*first2++);
      ret += dist * dist;
    }
    return ret > 0.0 ? sqrt(ret) : 0.0;
  }
};

class SingleTransitionCompleteVMO
{
public:
  int first_state_ = 0; /**< denotes the first state of the transition */
  int last_state_ = 0;  /**< denotes the last state of the transition */
  std::vector<double> vector_real_
      = {}; /**< denotes the feature std::vector of the transition */
  int corresponding_state_ = 0;
  int starting_frame_ = 0;
};

/** The class State denotes the elements that belong to each state
 * state denotes de number of the state
 * std::vector <SingleTransition> transition is the std::vector where all forward links of the state are defined
 * suffix_transition denotes which is the suffix link of this state
 * lrs is the longest repeated subsequence of this state
 * */
class StateCompleteVMO
{
public:
  int state_ = 0; /*!< denotes the number of the state */
  std::vector<SingleTransitionCompleteVMO> transition_
      = {}; /*!< denotes the number of the state */
  int suffix_transition_ = -1;
  int lrs_ = 0;
  int starting_frame_ = 0;
};

class Cluster
{
public:
  int label_ = 0;
  std::vector<int> Cluster = {};
};

class Clustering
{
public:
  std::vector<Cluster> Clustering = {};
};

class FilteredTransitionComplete
{
public:
  int current_lrs_;
  SingleTransitionCompleteVMO
      current_transition_; /**< denotes the transition */
};

class L
{
public:
  int state;
  std::vector<int> following_states_;
};

class VariableMarkovOracle
{
public:
  int pi_1 = 0, pi_2 = 0, k = 0, k_prime = 0, fo_iter = 0, M = 0,
      minimum_state = 0, counter_1 = 0;
  double MAX = 0, MIN = std::numeric_limits<double>::infinity(),
         feature_threshold = 0;
  std::vector<StateCompleteVMO> states_ = {}; /**< vector of all the states */
  std::vector<std::vector<int>> T
      = {}; /**< vector where each position has all the suffix transitions directed to each state */
  std::vector<std::vector<double>> S = {};
  std::map<int, double*> feature_map;
  Clustering clustering_;
  void AddFrame(
      int i,
      std::vector<double> vector_real,
      double threshold,
      int start_frame,
      int hop_size)
  {
    //! A normal member taking five arguments and returning no value.
    /*!
      \param i an integer argument that defines the current state.
      \param vector_real a vector of double values which contains the extracted feature values of the current state.
      \param threshold a double value which determines the level of similarity between vectors.
      \param start_frame an integer value which determines the start frame of the state.
      \param hop_size the hop_size that defines that frame length.
    */
    auto* context = new ContextVMO(new StrategyVMOEuclidean);
    this->CreateState(i + 1);
    int q_i_plus_one = 0, state_i_plus_one = i + 1;
    InitializeCluster(state_i_plus_one);
    this->states_[state_i_plus_one].suffix_transition_ = 0;
    this->T.emplace_back();
    this->AddTransition(
        i,
        state_i_plus_one,
        vector_real,
        q_i_plus_one,
        (state_i_plus_one + 1) * hop_size);
    k = this->states_[i].suffix_transition_; // < k = S[i]
    this->AddState(state_i_plus_one, 0, start_frame);
    pi_1 = i; //<  phi_one = i
    int iter = 0, counter = 0, s;
    bool flag = false;
    while (k > -1 && !flag)
    {
      iter = 0;
      bool less_distance_found = false;
      double minimum_euclidean_result
          = std::numeric_limits<double>::infinity();
      while (k > -1 && flag == 0)
      {
        iter = 0;
        minimum_euclidean_result = std::numeric_limits<double>::infinity();
        while (k > -1 && flag == 0)
        {
          if (iter < this->states_[k].transition_.size())
          {
            double* v2_pointer
                = &(this->states_[k].transition_[iter].vector_real_[0]);
            double* v1_pointer = &(vector_real[0]);
            int len_vector = vector_real.size();
            double euclidean_result = context->DoDistanceLogic(
                v1_pointer, (v1_pointer + (len_vector)), v2_pointer);
            if (euclidean_result < threshold
                && euclidean_result < minimum_euclidean_result)
            {
              minimum_state = iter;
              less_distance_found = true;
              minimum_euclidean_result = euclidean_result;
            }
            iter++;
            if (k == -1)
              flag = true;
            else
            {
              if (iter >= this->states_[k].transition_.size())
                flag = true;
            }
          }
        }
      }
      if (!less_distance_found)
      {
        AddTransition(
            k,
            state_i_plus_one,
            vector_real,
            q_i_plus_one,
            (state_i_plus_one + 1) * hop_size);
        k = this->states_[k].suffix_transition_;
      }
      if (less_distance_found)
      {
        k_prime = this->states_[k].transition_[minimum_state].last_state_;
        this->states_[state_i_plus_one].suffix_transition_ = k_prime;
        flag = true;
        break;
      }
      if (k == -1 || !flag)
      {
        flag = true;
      }
    }
    if (k == -1)
    {
      this->states_[state_i_plus_one].suffix_transition_ = 0;

      this->clustering_.Clustering[state_i_plus_one].Cluster.push_back(
          state_i_plus_one);
      this->clustering_.Clustering[state_i_plus_one].label_ = M + 1;
      M = M + 1;
    }
    else
    {
      int current_label = FindLabelClustering();
      // If element was found
      this->clustering_.Clustering[state_i_plus_one].label_ = current_label;
      UpdateCluster(
          state_i_plus_one,
          this->clustering_.Clustering[current_label].label_);
    }
    this->T[this->states_[state_i_plus_one].suffix_transition_].push_back(
        state_i_plus_one);
  };
  void CreateState(int m)
  {
    StateCompleteVMO new_state;
    new_state.state_ = m;
    new_state.transition_ = {};
    this->states_.push_back(new_state);
  };
  void SelectFeature(
      std::vector<std::vector<double>> audioFrame,
      int hop_size,
      std::string feature)
  {
    if (feature == "spectralCentroid")
    {
      this->SpectralCentroidFeatureExtraction(audioFrame, hop_size);
    }
    if (feature == "spectralRolloff")
    {
      this->SpectralRolloffFeatureExtraction(audioFrame, hop_size);
    }
    if (feature == "mfcc")
    {
      this->MFCCFeatureExtraction(audioFrame, hop_size);
    }
    if (feature == "rms")
    {
      this->RootMeanSquareFeatureExtraction(audioFrame, hop_size);
    }
    if (feature == "zeroCrossings")
    {
      this->ZeroCrossingsFeatureExtraction(audioFrame, hop_size);
    }
    if (feature == "pitch")
    {
      this->PitchFeatureExtraction(audioFrame, hop_size);
    }
  }
  void AnalyseAudio(
      std::string sfName,
      int hop_size,
      std::string feature,
      double threshold)
  {
    auto* librarySubsystem = new LibrarySubsystemVMO;
    auto* facade = new FacadeVMO(librarySubsystem);
    std::vector<std::vector<double>> audioFrame;
    feature_threshold = threshold;
    audioFrame = ClientCodeFramesVMO(facade, std::move(sfName), hop_size);
    this->AddInitialTransition();
    SelectFeature(audioFrame, hop_size, feature);
    int counter = this->states_.size();
  };
  int LengthCommonSuffix(int phi_one, int phi_two)
  {
    if (phi_two == this->states_[phi_one].suffix_transition_)
      return this->states_[phi_one].lrs_;
    else
    {
      while (this->states_[phi_one].suffix_transition_
             != this->states_[phi_two].suffix_transition_)
        phi_two = this->states_[phi_two].suffix_transition_;
    }
    return std::min(this->states_[phi_one].lrs_, this->states_[phi_two].lrs_)
           + 1;
  };
  int FindLabelClustering()
  {
    return this->clustering_.Clustering[k_prime].label_;
  }
  std::vector<int> VMOImprovisationList(int p)
  {
    std::vector<int> vector_l = {};
    if (p < this->clustering_.Clustering.size())
    {
      int j = 0;
      while (j < this->clustering_.Clustering.size())
      {
        if (this->clustering_.Clustering[j].label_ == p)
        {
          for (int& w : this->clustering_.Clustering[j].Cluster)
          {
            if (std::find(vector_l.begin(), vector_l.end(), w)
                != vector_l.end())
            {
              /* v contains x */
            }
            else
            {
              /* v does not contain x */
              vector_l.push_back(w);
            }
          }
        }
        j++;
      }
    }
    return vector_l;
  };
  std::vector<int> VMOGenerate(int i, int total_length, float q, int k)
  {
    ///! A normal member taking four arguments and returning a std::string value.
    /*!
           \param i an integer argument.
           \param v a std::string argument.
           \param q a float argument.
           \return The factor oracle improvisation
    */
    std::vector<int> improvisation_vector;
    int total_automata = this->states_.size();
    int iter = 0;
    int impro_jump = ceil(total_automata * 0.1);
    int impro_jump_two = ceil(total_automata * 0.075);
    int impro_jump_third = ceil(total_automata * 0.05);
    if (impro_jump > total_length)
    {
      impro_jump = ceil(impro_jump * 0.1);
      impro_jump_two = ceil(total_automata * 0.075);
      impro_jump_third = ceil(total_automata * 0.05);
    }
    int first_impro_jump = impro_jump;
    std::vector<int> impro_jumps;
    impro_jumps.push_back(first_impro_jump);
    impro_jumps.push_back(impro_jump_two);
    impro_jumps.push_back(impro_jump_third);
    for (iter = 0; iter < total_length; iter++)
    {
      std::vector<int> L = VMOImprovisationList(i);
      if (this->states_.size() == 2)
      {
        improvisation_vector.push_back(this->states_[0].state_);
      }
      else
      {
        if (iter == impro_jump)
        {
          std::random_device
              rd; //Will be used to obtain a seed for the random number engine
          std::mt19937 gen(
              rd()); //Standard mersenne_twister_engine seeded with rd()
          std::uniform_real_distribution<> dis(0, this->states_.size() - 1);
          int rand = dis(gen);
          i = rand; // estaba sacando una excepción porque el estado 103 era el último y no tenía starting frame. REVISAR EN EL FACTOR ORACLE
          improvisation_vector.push_back(this->states_[i].state_);
          std::random_device
              r; //Will be used to obtain a seed for the random number engine
          std::mt19937 gens(
              r()); //Standard mersenne_twister_engine seeded with rd()
          std::uniform_real_distribution<> dif(0, 2);
          int impro_value = dif(gens);
          impro_jump = impro_jump + impro_jumps[impro_value];
        }
        else
        {
          if (L.empty()) //Preguntar si debe iniciar en 1 o en 0
          {
            int len = this->states_.size();
            if (i >= len - 1)
            {
              std::random_device
                  rd; //Will be used to obtain a seed for the random number engine
              std::mt19937 gen(
                  rd()); //Standard mersenne_twister_engine seeded with rd()
              std::uniform_real_distribution<> dis(
                  0, this->states_.size() - 1);
              int rand = dis(gen);
              i = rand; // estaba sacando una excepción porque el estado 103 era el último y no tenía starting frame. REVISAR EN EL FACTOR ORACLE
              improvisation_vector.push_back(this->states_[i].state_);
            }
            else
            {
              improvisation_vector.push_back(
                  this->states_[i].transition_[0].last_state_);
            }
            i++;
          }
          else
          {
            if (i == states_.size())
              i = i - 1;
            int lenSuffix = L.size() - 1;
            std::random_device
                rd; //Will be used to obtain a seed for the random number engine
            std::mt19937 gen(
                rd()); //Standard mersenne_twister_engine seeded with rd()
            std::uniform_real_distribution<> dis(0, lenSuffix);
            int ran = dis(gen);
            improvisation_vector.push_back(L[ran]);
            i = L[ran];
            //i = this->states_[this->states_[i].suffix_transition_].transition_[ran].last_state_;
            if (i == -1)
              i = 0;
          }
        }
      }
    }
    return improvisation_vector;
  };
  void GenerateAudio(
      int i,
      int total_length,
      float q,
      int k_interal,
      int hop_size,
      int buffer_size,
      std::string input_filename,
      std::string output_filename)
  {
    AudioFile<double> audioFile;
    audioFile.load(std::move(input_filename));
    AudioFile<double>::AudioBuffer buffer;
    AudioFile<double>::AudioBuffer window_buffer;
    AudioFile<double>::AudioBuffer complete_buffer;
    int automaton_size = this->states_.size();
    int repetitions = (automaton_size * hop_size);
    int numChannels = audioFile.getNumChannels();
    int j = 0;
    std::vector<std::vector<double>> buffer_matrix;
    std::vector<double> sample;
    while (j < repetitions)
    {
      int j_temp = j;
      sample = {};
      int temp_final = j_temp + buffer_size;
      while (j_temp < temp_final)
      {
        sample.push_back(audioFile.samples[0][j_temp]);
        j_temp++;
      }
      buffer_matrix.push_back(sample);
      j = j + hop_size;
    }
    /*    for(int f = 0; f < total_length; f++){
        for(int g = 0; g < buffer_size; g++)
    }*/
    std::vector<int> improvisation
        = VMOGenerate(i, total_length, q, k_interal);

    // 2. Set to (e.g.) two channels
    buffer.resize(2);
    // repetitions = (total_length * buffer_size);
    repetitions = (total_length * hop_size);
    // 3. Set number of samples per channel
    buffer[0].resize(repetitions);
    buffer[1].resize(repetitions);

    // 2. Set to (e.g.) two channels
    complete_buffer.resize(2);

    // 3. Set number of samples per channel
    complete_buffer[0].resize(repetitions);
    complete_buffer[1].resize(repetitions);

    window_buffer.resize(2);
    window_buffer[0].resize(repetitions);
    window_buffer[1].resize(repetitions);

    std::vector<double> win = MakeWindow(buffer_size);
    int iter = 0;
    int real_iter = 0;
    repetitions = (total_length * hop_size);
    while (iter < repetitions)
    {
      //z = this->states_[improvisation[real_iter]].starting_frame_;
      for (int z = 0; z < buffer_size; z++)
      {
        for (int channel = 0; channel < numChannels; channel++)
          buffer[channel][iter]
              = buffer[channel][iter]
                + (buffer_matrix[improvisation[real_iter]][z] * win[z])
                + 0.00001;
        iter++;
      }
      if (iter < repetitions)
      {
        iter = iter - (hop_size);
        real_iter++;
      }
    }
    iter = 0;

    while (iter < repetitions)
    {
      //z = this->states_[improvisation[real_iter]].starting_frame_;

      for (int z = 0; z < buffer_size; z++)
      {
        for (int channel = 0; channel < numChannels; channel++)
          window_buffer[channel][iter] = window_buffer[channel][iter] + win[z];
        iter++;
      }
      if (iter == repetitions)
        break;
      if (iter < repetitions)
      {
        iter = iter - (hop_size);
      }
    }
    iter = 0;
    while (iter < repetitions)
    {
      //z = this->states_[improvisation[real_iter]].starting_frame_;

      for (int z = 0; z < buffer_size; z++)
      {
        for (int channel = 0; channel < numChannels; channel++)
          buffer[channel][iter]
              = buffer[channel][iter] / window_buffer[channel][iter];
        iter++;
        if (iter == repetitions)
          break;
      }
    }

    bool ok = audioFile.setAudioBuffer(buffer);
    // audioFile.setSampleRate (44100);
    audioFile.save(std::move(output_filename), AudioFileFormat::Wave);
  };
  static std::vector<double> MakeWindow(int n)
  {
    std::vector<double> window;
    window.reserve(n);
    for (int i = 0; i < n; i++)
      window.push_back((0.5 * (1 - cos(2 * M_PI * i / (n - 1)))) + 0.00001);
    return window;
  };
  static double VectorDistance(double* first, double* last, double* first2)
  {
    double ret = 0.0;
    while (first != last)
    {
      double dist = (*first++) - (*first2++);
      ret += dist * dist;
    }
    return ret > 0.0 ? sqrt(ret) : 0.0;
  };
  void FindBetter(
      std::vector<double> vector_real,
      int state_i_plus_one,
      int hop_size)
  {
    auto* context = new ContextVMO(new StrategyVMOEuclidean);
    int s = 0;
    int flag = 0, iter = 0;
    double min_distance = INFINITY;
    while (iter < this->states_[k].transition_.size())
    {

      double* v1_pointer = &(vector_real[0]);
      double* v2_pointer
          = &(this->states_[k].transition_[iter].vector_real_[0]);
      int len_vector = vector_real.size();
      double euclidean_result = context->DoDistanceLogic(
          v1_pointer, (v1_pointer + (len_vector)), v2_pointer);

      if (euclidean_result < min_distance)
      {
        s = this->states_[k].transition_[iter].last_state_;
        min_distance = euclidean_result;
      }
      iter++;
    }
    this->states_[state_i_plus_one].suffix_transition_ = s;
    this->states_[state_i_plus_one].lrs_ = this->LengthCommonSuffix(
        pi_1, this->states_[state_i_plus_one].suffix_transition_ - 1);
  };
  void AddState(int first_state, int state, int start_frame)
  {
    this->states_[first_state].suffix_transition_ = state;
    this->states_[first_state].lrs_ = state;
    this->states_[first_state].starting_frame_ = start_frame;
  };
  void AddTransition(
      int first_state,
      int last_state,
      std::vector<double> vector_real,
      int feature_state,
      int starting_frame)
  {
    SingleTransitionCompleteVMO transition_i;
    transition_i.first_state_ = first_state;
    transition_i.last_state_ = last_state;
    transition_i.vector_real_ = std::move(vector_real);
    transition_i.corresponding_state_ = feature_state;
    transition_i.starting_frame_ = starting_frame;
    this->states_[first_state].transition_.push_back(transition_i);

    counter_1 = counter_1 + 1;
  };
  void AddInitialTransition()
  {
    this->CreateState(0);
    this->states_[0].state_ = 0;
    this->states_[0].lrs_ = 0;
    this->states_[0].suffix_transition_ = -1;
    this->states_[0].starting_frame_ = 0;
    this->T.emplace_back();
  };
  void ZeroCrossingsFeatureExtraction(
      std::vector<std::vector<double>> audio_vector,
      int hop_size)
  {
    std::vector<std::vector<double>> vector_real;
    int numFrames = audio_vector.size();
    int counter = 0;
    int sampleRate = 44100;
    std::vector<double> temp_vector;
    std::vector<std::vector<double>> full_vector;
    temp_vector.reserve(1);
    for (int i = 0; i < numFrames; i++)
    {
      temp_vector = {};
      Gist<double> gist(hop_size, sampleRate);
      gist.processAudioFrame(audio_vector[i]);
      double zcr = gist.zeroCrossingRate();
      temp_vector.push_back(zcr);
      full_vector.push_back(temp_vector);
    }
    full_vector = NormalizeVectors(full_vector);
    InitializeCluster(0);
    for (int i = 0; i < numFrames; i++)
    {
      double* map_pointer = &(full_vector[i][0]);
      this->feature_map.insert({counter, map_pointer});
      this->AddFrame(
          counter, full_vector[i], feature_threshold, i * hop_size, hop_size);
      counter = counter + 1;
    }
  }
  void RootMeanSquareFeatureExtraction(
      std::vector<std::vector<double>> audio_vector,
      int hop_size)
  {
    std::vector<std::vector<double>> vector_real;
    int numFrames = audio_vector.size();
    int counter = 0;
    int frameSize = hop_size;
    int sampleRate = 44100;
    std::vector<double> temp_vector;
    std::vector<std::vector<double>> full_vector;
    temp_vector.reserve(1);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < numFrames; i++)
    {
      temp_vector = {};
      Gist<double> gist(frameSize, sampleRate);
      gist.processAudioFrame(audio_vector[i]);
      double rms = gist.rootMeanSquare();
      temp_vector.push_back(rms);
      full_vector.push_back(temp_vector);
    }
    full_vector = NormalizeVectors(full_vector);
    InitializeCluster(0);
    for (int i = 0; i < numFrames; i++)
    {
      double* map_pointer = &(full_vector[i][0]);
      this->feature_map.insert({counter, map_pointer});
      this->AddFrame(
          counter, full_vector[i], feature_threshold, i * hop_size, hop_size);
      counter = counter + 1;
    }
    auto end = std::chrono::steady_clock::now();
  }
  void MFCCFeatureExtraction(
      std::vector<std::vector<double>> audio_vector,
      int hop_size)
  {
    std::vector<std::vector<double>> vector_real;
    int numFrames = audio_vector.size();
    int counter = 0;
    int frameSize = hop_size;
    int sampleRate = 44100;
    std::vector<double> temp_vector;
    std::vector<std::vector<double>> full_vector;
    for (int i = 0; i < numFrames; i++)
    {
      temp_vector = {};
      Gist<double> gist(frameSize, sampleRate);
      gist.processAudioFrame(audio_vector[i]);
      const std::vector<double>& mfcc
          = gist.getMelFrequencyCepstralCoefficients();
      std::vector<double> mfcc_temp = mfcc;
      for (double j : mfcc_temp)
        /*            double * map_pointer = &(mfcc_temp[0]);
            this->feature_map.insert({counter, map_pointer});
            this->AddFrame(counter, mfcc_temp, 500 , i * hop_size, hop_size);
            counter = counter + 1;*/
        full_vector.push_back(mfcc_temp);
    }
    full_vector = NormalizeVectors(full_vector);
    for (const std::vector<double>& j : full_vector)
    {
      for (double w : j)
      {
      }
    }
    InitializeCluster(0);
    for (int i = 0; i < numFrames; i++)
    {
      double* map_pointer = &(full_vector[i][0]);
      this->feature_map.insert({counter, map_pointer});
      this->AddFrame(
          counter, full_vector[i], feature_threshold, i * hop_size, hop_size);
      counter = counter + 1;
    }
  }
  void PitchFeatureExtraction(
      std::vector<std::vector<double>> audio_vector,
      int hop_size)
  {
    std::vector<std::vector<double>> vector_real;
    int numFrames = audio_vector.size();
    int counter = 0;
    int frameSize = hop_size;
    int sampleRate = 44100;
    std::vector<double> temp_vector;
    std::vector<std::vector<double>> full_vector;
    temp_vector.reserve(1);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < numFrames; i++)
    {
      temp_vector = {};
      Gist<double> gist(frameSize, sampleRate);
      gist.processAudioFrame(audio_vector[i]);
      double pitch = gist.pitch();
      temp_vector.push_back(pitch);
      full_vector.push_back(temp_vector);
    }
    full_vector = NormalizeVectors(full_vector);
    InitializeCluster(0);
    for (int i = 0; i < numFrames; i++)
    {
      double* map_pointer = &(full_vector[i][0]);
      this->feature_map.insert({counter, map_pointer});
      this->AddFrame(
          counter, full_vector[i], feature_threshold, i * hop_size, hop_size);
      counter = counter + 1;
    }
    auto end = std::chrono::steady_clock::now();
  }
  void SpectralRolloffFeatureExtraction(
      std::vector<std::vector<double>> audio_vector,
      int hop_size)
  {
    std::vector<std::vector<double>> vector_real;
    int numFrames = audio_vector.size();
    int counter = 0;
    int frameSize = hop_size;
    int sampleRate = 44100;
    std::vector<double> temp_vector;
    std::vector<std::vector<double>> full_vector;
    temp_vector.reserve(1);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < numFrames; i++)
    {
      temp_vector = {};
      Gist<double> gist(frameSize, sampleRate);
      gist.processAudioFrame(audio_vector[i]);
      double spectralRolloff = gist.spectralRolloff();
      temp_vector.push_back(spectralRolloff);
      full_vector.push_back(temp_vector);
    }
    full_vector = NormalizeVectors(full_vector);
    InitializeCluster(0);
    for (int i = 0; i < numFrames; i++)
    {
      double* map_pointer = &(full_vector[i][0]);
      this->feature_map.insert({counter, map_pointer});
      this->AddFrame(
          counter, full_vector[i], feature_threshold, i * hop_size, hop_size);
      counter = counter + 1;
    }
    auto end = std::chrono::steady_clock::now();
  }
  void SpectralCentroidFeatureExtraction(
      std::vector<std::vector<double>> audio_vector,
      int hop_size)
  {
    std::vector<std::vector<double>> vector_real;
    int numFrames = audio_vector.size();
    int counter = 0;
    int frameSize = hop_size;
    int sampleRate = 44100;
    std::vector<double> temp_vector;
    std::vector<std::vector<double>> full_vector;
    temp_vector.reserve(1);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < numFrames; i++)
    {
      temp_vector = {};
      Gist<double> gist(frameSize, sampleRate);
      gist.processAudioFrame(audio_vector[i]);
      double spectralCentroid = gist.spectralCentroid();
      temp_vector.push_back(spectralCentroid);
      full_vector.push_back(temp_vector);
    }
    full_vector = NormalizeVectors(full_vector);
    for (std::vector<double> j : full_vector)
    {
      for (double w : j)
      {
      }
    }
    InitializeCluster(0);
    for (int i = 0; i < numFrames; i++)
    {
      double* map_pointer = &(full_vector[i][0]);
      this->feature_map.insert({counter, map_pointer});
      this->AddFrame(
          counter, full_vector[i], feature_threshold, i * hop_size, hop_size);
      counter = counter + 1;
    }
    auto end = std::chrono::steady_clock::now();
  }
  std::vector<std::vector<double>>
  NormalizeVectors(std::vector<std::vector<double>> feature_vector)
  {
    for (const std::vector<double>& current_vector : feature_vector)
    {
      FindMaxAndMin(current_vector);
    }
    for (std::vector<double>& current_vector : feature_vector)
    {
      current_vector = NormalizeValues(current_vector);
    }
    for (std::vector<double> j : feature_vector)
    {
      for (double w : j)
      {
      }
    }
    return feature_vector;
  }
  void FindMaxAndMin(const std::vector<double>& vector_numbers)
  {
    for (double j : vector_numbers)
    {
      if (j < MIN)
        MIN = j;
      if (j > MAX)
        MAX = j;
    }
  }
  std::vector<double> NormalizeValues(std::vector<double> feature_vector) const
  {
    for (double& i : feature_vector)
    {
      i = (i - MIN) / (MAX - MIN);
    }
    return feature_vector;
  }
  void InitializeClusterAfter(int frame_index)
  {
    Cluster new_cluster;
    new_cluster.label_ = M + 1;
    new_cluster.Cluster.push_back(frame_index);
    this->clustering_.Clustering.push_back(new_cluster);
  }
  void InitializeCluster(int frame_index)
  {
    Cluster new_cluster;
    new_cluster.label_ = 0;
    //new_cluster.Cluster.push_back(frame_index);
    this->clustering_.Clustering.push_back(new_cluster);
  }
  void UpdateCluster(int frame_index, int suffix_index)
  {
    this->clustering_.Clustering[suffix_index].Cluster.push_back(frame_index);
  }
};

void GainDoubler::apply(
    Process::ProcessModel& proc,
    const score::DocumentContext& ctx)

{
  auto& sound = safe_cast<Media::Sound::ProcessModel&>(proc);
  auto& file = sound.file();

  // Current file path:
  qDebug() << file->absoluteFileName();
  std::string fileName = file->absoluteFileName().toStdString();
  std::vector<std::string> v;
  std::string temp = "";
  std::istringstream ss(fileName);
  std::string token;
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
  auto str = oss.str();

  std::string current_path = std::filesystem::current_path();
  std::string temp_path
      = current_path.append("/Processed/newAudioVMOImpro_" + str + ".wav");
  QString new_path = QString::fromUtf8(temp_path.c_str());
  // Get the data: (note: this performs a copy and could be slow for large files)
  auto array = file->getAudioArray();

  // Perform our offline processing
  VariableMarkovOracle variable_markov_oracle;
  QProgressDialog progress(
      "Processing...", "In Progress", 0, 100, qApp->activeWindow());
  progress.setWindowModality(Qt::WindowModal);
  progress.open();
  variable_markov_oracle.AnalyseAudio(
      fileName, 16384, "spectralRolloff", 0.0005);
  std::random_device
      rd; //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(
      0, variable_markov_oracle.states_.size());
  int start_state = dis(gen);
  variable_markov_oracle.GenerateAudio(
      start_state,
      variable_markov_oracle.states_.size(),
      0.8,
      0,
      16384,
      32768,
      fileName,
      "./Processed/newAudioVMOImpro_" + str + ".wav");
  // 16384
  progress.setValue(100);
  progress.close();
  AudioFile<double> audio_file;
  audio_file.load(temp_path);

  for (int i = 0; i < array.size(); i++)
  {
    for (int j = 0; j < array[i].size(); j++)
    {
      array[i][j] = audio_file.samples[0][j];
    }
  }
  // Check if we have a project folder
  if (auto path = ctx.document.metadata().projectFolder(); !path.isEmpty())
  {
    // Get a new file name in the project folder
    auto newFilename = score::newProcessedFilePath(
        file->absoluteFileName(),
        QDir(path + QDir::separator() + "Processed"));
    qDebug() << newFilename;

    // Create the path if necessary
    QDir{path}.mkpath(QFileInfo(newFilename).absolutePath());

    // Save the data in the .wav file
    Media::writeAudioArrayToFile(newFilename, array, file->sampleRate());

    // Tell the process to load the new file
    CommandDispatcher<> cmd{ctx.commandStack};
    cmd.submit<Media::LoadProcessedAudioFile>(sound, newFilename);
  }
  else
  {
    qDebug() << "Could not save processed file ! "
                "You must save the score project somewhere beforehand.";
  }
}
}
