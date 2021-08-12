#pragma once
#include <Media/Commands/ChangeAudioFile.hpp>
#include <Media/Sound/SoundModel.hpp>
#include <Process/OfflineAction/OfflineAction.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>
#include <score/document/DocumentContext.hpp>

#include <core/document/Document.hpp>

#include <ossia/audio/drwav_handle.hpp>

#include <QDir>
namespace AudioOracleRMS
{

class GainDoubler final : public Process::OfflineAction
{
  SCORE_CONCRETE("AE7A1FD1-A6CA-4869-A67F-41F255216264")

  QString title() const noexcept override;
  UuidKey<Process::ProcessModel> target() const noexcept override;
  void
  apply(Process::ProcessModel& proc, const score::DocumentContext&) override;
};
}