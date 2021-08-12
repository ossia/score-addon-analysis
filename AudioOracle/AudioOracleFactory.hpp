#pragma once
#include <Process/OfflineAction/OfflineAction.hpp>
#include <Media/Sound/SoundModel.hpp>
#include <Media/Commands/ChangeAudioFile.hpp>

#include <score/document/DocumentContext.hpp>
#include <score/command/Dispatchers/CommandDispatcher.hpp>

#include <core/document/Document.hpp>

#include <ossia/audio/drwav_handle.hpp>

#include <QDir>
namespace AudioOracle
{

class GainDoubler final : public Process::OfflineAction
{
  SCORE_CONCRETE("d1d8f66e-6f85-46ff-b8fb-e207db2cb8a2")

  QString title() const noexcept override;
  UuidKey<Process::ProcessModel> target() const noexcept override;
  void apply(Process::ProcessModel& proc, const score::DocumentContext&) override;
};
}