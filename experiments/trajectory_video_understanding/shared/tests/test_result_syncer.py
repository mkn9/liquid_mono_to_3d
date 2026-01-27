"""
Tests for MacBookSyncer
========================

TDD tests for result synchronization functionality.
Tests are written BEFORE implementation (RED phase).
"""

import pytest
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, call
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from result_syncer import MacBookSyncer


class TestMacBookSyncerInit:
    """Test syncer initialization."""
    
    def test_syncer_creation(self):
        """Test basic syncer creation."""
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path'
        )
        
        assert syncer.worker_name == 'test_worker'
        assert syncer.macbook_path == 'user@host:/path'
        assert syncer.enabled is True
    
    def test_syncer_disabled(self):
        """Test syncer can be disabled."""
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path',
            enabled=False
        )
        
        assert syncer.enabled is False


class TestHeartbeatPushing:
    """Test heartbeat functionality."""
    
    def test_heartbeat_creates_file(self):
        """Test that heartbeat creates a JSON file locally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir).joinpath('heartbeat.json').unlink(missing_ok=True)
            
            syncer = MacBookSyncer(
                worker_name='test_worker',
                macbook_path='user@host:/path',
                enabled=True  # Enable to test file creation
            )
            
            # Push heartbeat
            with patch.object(syncer, '_scp_file', return_value=True):
                syncer.push_heartbeat(epoch=1, step=100, loss=0.5)
            
            # Check file was created
            heartbeat_file = Path('heartbeat.json')
            assert heartbeat_file.exists()
            
            # Check content
            with open(heartbeat_file) as f:
                data = json.load(f)
            
            assert data['worker'] == 'test_worker'
            assert data['epoch'] == 1
            assert data['step'] == 100
            assert data['loss'] == 0.5
            assert data['status'] == 'training'
            assert 'timestamp' in data
            
            # Cleanup
            heartbeat_file.unlink(missing_ok=True)
    
    def test_heartbeat_calls_scp(self):
        """Test that heartbeat calls SCP when enabled."""
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path',
            enabled=True
        )
        
        with patch.object(syncer, '_scp_file', return_value=True) as mock_scp:
            result = syncer.push_heartbeat(epoch=1, step=100, loss=0.5)
            
            assert result is True
            mock_scp.assert_called_once()
            args = mock_scp.call_args[0]
            assert args[0] == Path('heartbeat.json')
            assert args[1] == 'heartbeat.json'
        
        # Cleanup
        Path('heartbeat.json').unlink(missing_ok=True)
    
    def test_heartbeat_disabled_no_push(self):
        """Test that disabled syncer doesn't push."""
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path',
            enabled=False
        )
        
        with patch.object(syncer, '_scp_file') as mock_scp:
            result = syncer.push_heartbeat(epoch=1, step=100, loss=0.5)
            
            assert result is False
            mock_scp.assert_not_called()


class TestCheckpointPushing:
    """Test checkpoint pushing functionality."""
    
    def test_checkpoint_creates_metrics_file(self):
        """Test that checkpoint pushing creates metrics file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint_epoch_2.pt'
            checkpoint_path.touch()
            
            syncer = MacBookSyncer(
                worker_name='test_worker',
                macbook_path='user@host:/path',
                enabled=True
            )
            
            metrics = {
                'train_loss': 0.3,
                'val_loss': 0.4,
                'val_accuracy': 0.85
            }
            
            with patch.object(syncer, '_scp_file', return_value=True):
                syncer.push_checkpoint(epoch=2, checkpoint_path=checkpoint_path, metrics=metrics)
            
            # Check metrics file was created
            metrics_file = Path('metrics_epoch_2.json')
            assert metrics_file.exists()
            
            with open(metrics_file) as f:
                data = json.load(f)
            
            assert data['train_loss'] == 0.3
            assert data['val_loss'] == 0.4
            assert data['val_accuracy'] == 0.85
            
            # Cleanup
            metrics_file.unlink(missing_ok=True)
    
    def test_checkpoint_pushes_both_files(self):
        """Test that checkpoint pushing calls SCP for both checkpoint and metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint_epoch_2.pt'
            checkpoint_path.touch()
            
            syncer = MacBookSyncer(
                worker_name='test_worker',
                macbook_path='user@host:/path',
                enabled=True
            )
            
            metrics = {'train_loss': 0.3}
            
            with patch.object(syncer, '_scp_file', return_value=True) as mock_scp:
                result = syncer.push_checkpoint(epoch=2, checkpoint_path=checkpoint_path, metrics=metrics)
                
                assert result is True
                assert mock_scp.call_count == 2
                
                # Check both files were pushed
                calls = [call[0] for call in mock_scp.call_args_list]
                filenames = [c[1] for c in calls]
                assert 'metrics_epoch_2.json' in filenames
                assert 'checkpoint_epoch_2.pt' in filenames
            
            # Cleanup
            Path('metrics_epoch_2.json').unlink(missing_ok=True)


class TestLogTailing:
    """Test log tailing functionality."""
    
    def test_log_tail_creates_file(self):
        """Test that log tailing creates a tail file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'train.log'
            
            # Write 100 lines to log
            with open(log_file, 'w') as f:
                for i in range(100):
                    f.write(f"Log line {i}\n")
            
            syncer = MacBookSyncer(
                worker_name='test_worker',
                macbook_path='user@host:/path',
                enabled=True
            )
            
            with patch.object(syncer, '_scp_file', return_value=True):
                syncer.push_log_tail(log_file, num_lines=10)
            
            # Check tail file was created
            tail_file = Path('latest_log.txt')
            assert tail_file.exists()
            
            # Check it has last 10 lines
            with open(tail_file) as f:
                lines = f.readlines()
            
            assert len(lines) == 10
            assert lines[0] == "Log line 90\n"
            assert lines[-1] == "Log line 99\n"
            
            # Cleanup
            tail_file.unlink(missing_ok=True)
    
    def test_log_tail_nonexistent_file(self):
        """Test that log tailing handles nonexistent files."""
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path',
            enabled=True
        )
        
        result = syncer.push_log_tail(Path('nonexistent.log'))
        assert result is False


class TestCompletionStatus:
    """Test training completion status."""
    
    def test_completion_status_success(self):
        """Test completion status for successful training."""
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path',
            enabled=True
        )
        
        final_metrics = {
            'final_loss': 0.1,
            'final_accuracy': 0.95
        }
        
        with patch.object(syncer, '_scp_file', return_value=True):
            syncer.push_completion_status(success=True, final_metrics=final_metrics)
        
        status_file = Path('training_status.json')
        assert status_file.exists()
        
        with open(status_file) as f:
            data = json.load(f)
        
        assert data['status'] == 'completed'
        assert data['worker'] == 'test_worker'
        assert data['final_metrics']['final_loss'] == 0.1
        
        # Cleanup
        status_file.unlink(missing_ok=True)
    
    def test_completion_status_failure(self):
        """Test completion status for failed training."""
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path',
            enabled=True
        )
        
        with patch.object(syncer, '_scp_file', return_value=True):
            syncer.push_completion_status(success=False, final_metrics={})
        
        status_file = Path('training_status.json')
        with open(status_file) as f:
            data = json.load(f)
        
        assert data['status'] == 'failed'
        
        # Cleanup
        status_file.unlink(missing_ok=True)


class TestSCPIntegration:
    """Test SCP file transfer integration."""
    
    @patch('subprocess.run')
    def test_scp_called_with_correct_args(self, mock_run):
        """Test that SCP is called with correct arguments."""
        mock_run.return_value = Mock(returncode=0)
        
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@macbook:/remote/path',
            ssh_key='/path/to/key.pem',
            enabled=True
        )
        
        test_file = Path('test.txt')
        test_file.write_text('test')
        
        result = syncer._scp_file(test_file, 'test.txt')
        
        assert result is True
        mock_run.assert_called_once()
        
        # Check subprocess args
        args = mock_run.call_args[0][0]
        assert args[0] == 'scp'
        assert '/path/to/key.pem' in args
        assert 'user@macbook:/remote/path/test_worker/test.txt' in args
        
        # Cleanup
        test_file.unlink(missing_ok=True)
    
    @patch('subprocess.run')
    def test_scp_handles_failure(self, mock_run):
        """Test that SCP failures are handled gracefully."""
        mock_run.return_value = Mock(returncode=1)
        
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path',
            enabled=True
        )
        
        test_file = Path('test.txt')
        test_file.write_text('test')
        
        result = syncer._scp_file(test_file, 'test.txt')
        
        assert result is False
        
        # Cleanup
        test_file.unlink(missing_ok=True)
    
    @patch('subprocess.run')
    def test_scp_handles_timeout(self, mock_run):
        """Test that SCP timeouts are handled."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='scp', timeout=10)
        
        syncer = MacBookSyncer(
            worker_name='test_worker',
            macbook_path='user@host:/path',
            enabled=True
        )
        
        test_file = Path('test.txt')
        test_file.write_text('test')
        
        result = syncer._scp_file(test_file, 'test.txt')
        
        assert result is False
        
        # Cleanup
        test_file.unlink(missing_ok=True)

