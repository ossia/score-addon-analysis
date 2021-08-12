#include "AudioOracleMFCCFactory.hpp"

#include <Media/Commands/ChangeAudioFile.hpp>
#include <Media/Sound/SoundModel.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>
#include <score/document/DocumentContext.hpp>

#include <core/document/Document.hpp>

#include <ossia/audio/drwav_handle.hpp>

#include <QApplication>
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

namespace AudioOracleMFCC
{
QString GainDoubler::title() const noexcept
{
  return QObject::tr("Audio Oracle - MFCC");
}

UuidKey<Process::ProcessModel> GainDoubler::target() const noexcept
{
  return Metadata<ConcreteKey_k, Media::Sound::ProcessModel>::get();
}

class LibrarySubsystem
{
public:
  static std::vector<std::vector<double>>
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
 * The Facade class provides a simple interface to the complex logic of one or
 * several subsystems. The Facade delegates the client requests to the
 * appropriate objects within the subsystem. The Facade is also responsible for
 * managing their lifecycle. All of this shields the client from the undesired
 * complexity of the subsystem.
 */
class Facade
{
protected:
  LibrarySubsystem* librarySubsystem_;
  /**
   * Depending on your application's needs, you can provide the Facade with
   * existing subsystem objects or force the Facade to create them on its own.
   */
public:
  /**
   * In this case we will delegate the memory ownership to Facade Class
   */
  explicit Facade(LibrarySubsystem* librarySubsystem = nullptr)
  {
    this->librarySubsystem_ = librarySubsystem ?: new LibrarySubsystem;
  }
  ~Facade() { delete librarySubsystem_; }
  /**
   * The Facade's methods are convenient shortcuts to the sophisticated
   * functionality of the subsystems. However, clients get only to a fraction of
   * a subsystem's capabilities.
   */
  static std::vector<std::vector<double>>
  OperationGenerateFrames(std::string input_filename, int hop_size)
  {
    return LibrarySubsystem::GenerateAudioFrames(
        std::move(input_filename), hop_size);
  }
};

/**
 * The client code works with complex subsystems through a simple interface
 * provided by the Facade. When a facade manages the lifecycle of the subsystem,
 * the client might not even know about the existence of the subsystem. This
 * approach lets you keep the complexity under control.
 */

std::vector<std::vector<double>>
ClientCodeFrames(Facade* facade, std::string input_filename, int hop_size)
{
  return Facade::OperationGenerateFrames(std::move(input_filename), hop_size);
}
/**
 * The client code may have some of the subsystem's objects already created. In
 * this case, it might be worthwhile to initialize the Facade with these objects
 * instead of letting the Facade create new instances.
 */

/**
 * The Strategy interface declares operations common to all supported versions
 * of some algorithm.
 *
 * The Context uses this interface to call the algorithm defined by Concrete
 * Strategies.
 */
class Strategy
{
public:
  virtual ~Strategy() = default;
  virtual double
  DoVectorDistance(double* first, double* last, double* first2) const = 0;
};

/**
 * The Context defines the interface of interest to clients.
 */

class Context
{
  /**
   * @var Strategy The Context maintains a reference to one of the Strategy
   * objects. The Context does not know the concrete class of a strategy. It
   * should work with all strategies via the Strategy interface.
   */
private:
  Strategy* strategy_;
  /**
   * Usually, the Context accepts a strategy through the constructor, but also
   * provides a setter to change it at runtime.
   */
public:
  explicit Context(Strategy* strategy = nullptr)
      : strategy_(strategy)
  {
  }
  ~Context() { delete this->strategy_; }
  /**
   * Usually, the Context allows replacing a Strategy object at runtime.
   */
  void set_strategy(Strategy* strategy)
  {
    delete this->strategy_;
    this->strategy_ = strategy;
  }
  /**
   * The Context delegates some work to the Strategy object instead of
   * implementing +multiple versions of the algorithm on its own.
   */
  double DoDistanceLogic(double* first, double* last, double* first2) const
  {
    double result = this->strategy_->DoVectorDistance(first, last, first2);
    return result;
  }
};

/**
 * Concrete Strategies implement the algorithm while following the base Strategy
 * interface. The interface makes them interchangeable in the Context.
 */
class StrategyEuclidean : public Strategy
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
class ConcreteStrategyB : public Strategy
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

class SingleTransition
{
public:
  int first_state_{}; /**< denotes the first state of the transition */
  int last_state_{};  /**< denotes the last state of the transition */
  std::vector<double>
      vector_real_; /**< denotes the feature std::vector of the transition */
  int corresponding_state_{};
  int starting_frame_{};
};

/** The class State denotes the elements that belong to each state
 * state denotes de number of the state
 * std::vector <SingleTransition> transition is the std::vector where all forward links of the state are defined
 * suffix_transition denotes which is the suffix link of this state
 * lrs is the longest repeated subsequence of this state
 * */
class State
{
public:
  int state_{}; /*!< denotes the number of the state */
  std::vector<SingleTransition>
      transition_; /*!< denotes the number of the state */
  int suffix_transition_{};
  int lrs_ = 0;
  int starting_frame_ = 0;
};

class AudioOracle
{
public:
  int pi_1 = 0, pi_2 = 0, k = 0, fo_iter = 0;
  double MAX = 0, MIN = std::numeric_limits<double>::infinity(),
         feature_threshold = 0;
  std::vector<State> states_; /**< vector of all the states */
  std::vector<std::vector<int>>
      T; /**< vector where each position has all the suffix transitions directed to each state */
  std::map<int, double*> feature_map;
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
    auto* context = new Context(new StrategyEuclidean);
    this->CreateState(i + 1);
    int state_i_plus_one = i + 1;
    this->T.emplace_back();
    this->AddTransition(
        i,
        state_i_plus_one,
        vector_real,
        i,
        (state_i_plus_one + 1) * hop_size);
    k = this->states_[i].suffix_transition_; // < k = S[i]
    this->AddState(state_i_plus_one, 0, start_frame);
    pi_1 = i; //<  phi_one = i
    int flag = 0, iter = 0, counter = 0, s;
    while (k > -1 && flag == 0)
    {
      iter = 0;
      double minimum_euclidean_result
          = std::numeric_limits<double>::infinity();
      while (k > -1 && flag == 0)
      {
        if (iter < this->states_[k].transition_.size())
        {
          double* v2_pointer
              = &(this->states_[k].transition_[iter].vector_real_[0]);
          double* v1_pointer = &(vector_real[0]);
          iter++;
          int len_vector = vector_real.size();
          double euclidean_result = context->DoDistanceLogic(
              v1_pointer, (v1_pointer + (len_vector)), v2_pointer);
          if (euclidean_result < threshold)
          {
            AddTransition(
                k,
                state_i_plus_one,
                vector_real,
                i,
                (state_i_plus_one + 1) * hop_size);
            pi_1 = k;
            k = this->states_[k].suffix_transition_;
            minimum_euclidean_result = euclidean_result;
          }
          if (k == -1)
            flag = 1;
          else
          {
            if (iter >= this->states_[k].transition_.size())
              flag = 1;
          }
        }
      }
    }
    if (k == -1)
    {
      this->states_[state_i_plus_one].suffix_transition_ = 0;
      this->states_[state_i_plus_one].lrs_ = 0;
    }
    else
    {
      FindBetter(vector_real, state_i_plus_one, hop_size);
    }
    this->T[this->states_[state_i_plus_one].suffix_transition_].push_back(
        state_i_plus_one);
  };
  void CreateState(int m)
  {
    State newstate;
    newstate.state_ = m;
    this->states_.push_back(newstate);
  };
  void SelectFeature(
      const std::vector<std::vector<double>>& audioFrame,
      int hop_size,
      const std::string& feature)
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
      const std::string& feature,
      double threshold)
  {
    auto* librarySubsystem = new LibrarySubsystem;
    auto* facade = new Facade(librarySubsystem);
    std::vector<std::vector<double>> audioFrame;
    feature_threshold = threshold;
    audioFrame = ClientCodeFrames(facade, std::move(sfName), hop_size);
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
  std::vector<int> AOGenerate(int i, int total_length, float q, int k)
  {
    ///! A normal member taking four arguments and returning a std::string value.
    /*!
           \param i an integer argument.
           \param v a std::string argument.
           \param q a float argument.
           \return The factor oracle improvisation
    */
    std::vector<int> improvisation_vector;
    int iter = 0;
    for (iter = 0; iter < total_length; iter++)
    {
      std::random_device
          rd; //Will be used to obtain a seed for the random number engine
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);
      float u = dis(gen);
      if (this->states_.size() == 2)
      {
        improvisation_vector.push_back(this->states_[0].state_);
      }
      else
      {
        if (u < q)
        {
          int len = this->states_.size();
          if (i >= len - 1)
          {
            i = len - 1;
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
          int lenSuffix = this->states_[this->states_[i].suffix_transition_]
                              .transition_.size()
                          - 1;
          int rand_alpha = 0;
          if (lenSuffix == -1)
          {
            improvisation_vector.push_back(this->states_[i].state_);
            i++;
          }
          else
          {
            std::random_device
                rd; //Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis_int(0, lenSuffix);
            rand_alpha = dis_int(gen);
            improvisation_vector.push_back(
                this->states_[this->states_[i].suffix_transition_]
                    .transition_[rand_alpha]
                    .last_state_);
          }
          i = this->states_[this->states_[i].suffix_transition_]
                  .transition_[rand_alpha]
                  .last_state_;
          if (i == -1)
            i = 0;
        }
      }
    }
    return improvisation_vector;
  };
  ;
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
    std::vector<int> improvisation = AOGenerate(i, total_length, q, k_interal);

    for (int f = 0; f < total_length; f++)
    {
    }
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
    for (int f = 0; f < total_length; f++)
    {
    }
    int iter = 0;
    int real_iter = 0;
    repetitions = (total_length * hop_size);
    while (iter < repetitions)
    {
      for (int z = 0; z < buffer_size; z++)
      {
        for (int channel = 0; channel < numChannels; channel++)
          buffer[channel][iter]
              = buffer[channel][iter]
                + (buffer_matrix[improvisation[real_iter]][z] * win[z])
                + 0.00001;
        iter++;
        if (iter == repetitions)
          break;
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

      for (int z = 0; z < buffer_size; z++)
      {
        for (int channel = 0; channel < numChannels; channel++)
          window_buffer[channel][iter] = window_buffer[channel][iter] + win[z];
        iter++;
      }
      if (iter < repetitions)
      {
        iter = iter - (hop_size);
      }
    }
    iter = 0;
    while (iter < repetitions)
    {

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
  void FindBetter(
      std::vector<double> vector_real,
      int state_i_plus_one,
      int hop_size)
  {
    auto* context = new Context(new StrategyEuclidean);
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
    SingleTransition transition_i;
    transition_i.first_state_ = first_state;
    transition_i.last_state_ = last_state;
    transition_i.vector_real_ = std::move(vector_real);
    transition_i.corresponding_state_ = feature_state;
    transition_i.starting_frame_ = starting_frame;
    this->states_[first_state].transition_.push_back(transition_i);
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
        full_vector.push_back(mfcc_temp);
    }
    full_vector = NormalizeVectors(full_vector);
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
      = current_path.append("/Processed/newAudioImpro_" + str + ".wav");
  QString new_path = QString::fromUtf8(temp_path.c_str());
  // Get the data: (note: this performs a copy and could be slow for large files)
  auto array = file->getAudioArray();

  // Perform our offline processing
  AudioOracle audio_oracle;
  QProgressDialog progress(
      "Processing...", "In Progress", 0, 100, qApp->activeWindow());
  progress.setWindowModality(Qt::WindowModal);
  progress.open();
  //... copy one file
  audio_oracle.AnalyseAudio(fileName, 32768, "mfcc", 0.08);
  progress.setValue(50);
  std::random_device
      rd; //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, audio_oracle.states_.size());
  int start_state = dis(gen);
  audio_oracle.GenerateAudio(
      start_state,
      audio_oracle.states_.size(),
      0.8,
      0,
      32768,
      65536,
      fileName,
      "./Processed/newAudioImpro_" + str + ".wav");
  progress.setValue(100);
  // 16384
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
