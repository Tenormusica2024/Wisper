import warnings
import json
import time
import re
import os  # 追加：osモジュールのインポート
from datetime import datetime
from pathlib import Path
import tempfile
import subprocess
from typing import Dict, Optional, List

print("=== Whisper + Claude統合音声処理システム ===")
print("ライブラリ読み込み開始...")

# 基本ライブラリのインポート
try:
    import whisper
    import yt_dlp
    import numpy as np
    import requests
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    print("✅ 基本ライブラリ読み込み完了")
except ImportError as e:
    print(f"❌ 基本ライブラリエラー: {e}")
    exit(1)

warnings.filterwarnings("ignore")

class ClaudeWhisperIntegration:
    """Claude 3.5 Sonnet + Whisper統合処理クラス"""
    
    def __init__(self, claude_api_key: str = None):
        # 自己学習用辞書のパス
        self.dictionary_folder = "D:\\Python\\テキスト修正"
        self.learned_corrections_file = os.path.join(self.dictionary_folder, "learned_corrections.json")
        self.confidence_patterns_file = os.path.join(self.dictionary_folder, "confidence_patterns.json")
        
        # APIキー設定（複数の方法でロード）
        self.claude_api_key = self._load_api_key(claude_api_key)
        self.claude_model = "claude-3-5-sonnet-20241022"
        self.claude_base_url = "https://api.anthropic.com/v1/messages"
        
        # 辞書フォルダ作成
        os.makedirs(self.dictionary_folder, exist_ok=True)
        
        # 学習済み修正パターンを読み込み
        self.learned_corrections = self.load_learned_corrections()
        self.confidence_patterns = self.load_confidence_patterns()
        
        # Whisper特有の誤字パターン（初期辞書）
        self.whisper_patterns = {
            '地民党': '自民党', '石橋ロシ': '石破茂', '急安部派': '旧安倍派',
            '党生年局': '党執行部', '党主': '党首', '階段': '会談', '手動': '支持',
            'わわれな': '哀れな', '溜まるか': 'たまるか', '駄目に': 'だめに',
            '異見': '意見', '腐り切ってる': '腐りきっている',
            'ガソリンを': 'ガソリン税を', '販売担保': '販売価格', 'さっさに進化': '先走り',
            'ガソスタ': 'ガソリンスタンド'
        }
        
        # API使用回数とコスト追跡
        self.processing_stats = {
            'total_files': 0,
            'successful_transcriptions': 0,
            'claude_corrections': 0,
            'claude_classifications': 0,
            'total_cost': 0.0,
            'classification_cost': 0.0,
            'processing_time': 0.0,
            'learned_patterns': 0
        }
    
    def _load_api_key(self, provided_key: str = None) -> str:
        """APIキーを安全にロード（優先順位順）"""
        
        # 1. 引数で直接指定された場合
        if provided_key:
            print("🔑 APIキー: 引数から取得")
            return provided_key
        
        # 2. 環境変数から取得
        env_key = os.environ.get('CLAUDE_API_KEY')
        if env_key:
            print("🔑 APIキー: 環境変数から取得")
            return env_key
        
        # 3. 設定ファイルから取得
        config_file = os.path.join(self.dictionary_folder, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if config.get('claude_api_key'):
                        print("🔑 APIキー: config.jsonから取得")
                        return config['claude_api_key']
            except Exception as e:
                print(f"⚠️ config.json読み込みエラー: {e}")
        
        # 4. 暗号化ファイルから取得
        encrypted_file = os.path.join(self.dictionary_folder, "api_key.enc")
        if os.path.exists(encrypted_file):
            try:
                key = self._load_encrypted_api_key(encrypted_file)
                if key:
                    print("🔑 APIキー: 暗号化ファイルから取得")
                    return key
            except Exception as e:
                print(f"⚠️ 暗号化ファイル読み込みエラー: {e}")
        
        # 5. プレーンテキストファイルから取得（非推奨）
        plain_file = os.path.join(self.dictionary_folder, "api_key.txt")
        if os.path.exists(plain_file):
            try:
                with open(plain_file, 'r', encoding='utf-8') as f:
                    key = f.read().strip()
                    if key:
                        print("🔑 APIキー: api_key.txtから取得（⚠️ セキュリティ注意）")
                        print("💡 推奨: 暗号化ファイルまたは環境変数の使用")
                        return key
            except Exception as e:
                print(f"⚠️ api_key.txt読み込みエラー: {e}")
        
        # 6. 対話式入力
        print("🔑 APIキーが見つかりません。設定してください。")
        return self._setup_api_key_interactive()
    
    def _setup_api_key_interactive(self) -> str:
        """対話式APIキー設定"""
        print("\n🔧 === APIキー設定 ===")
        print("1. 🔒 暗号化ファイルに保存（推奨）")
        print("2. 📄 設定ファイル（JSON）に保存")
        print("3. 📝 プレーンテキストに保存（非推奨）")
        print("4. ⚠️ 今回のみ使用（保存しない）")
        
        try:
            choice = input("\n選択してください (1-4): ").strip()
            api_key = input("\n🔑 Claude APIキーを入力: ").strip()
            
            if not api_key:
                raise ValueError("APIキーが入力されていません")
            
            if choice == '1':
                self._save_encrypted_api_key(api_key)
                print("✅ 暗号化ファイルに保存しました")
            
            elif choice == '2':
                self._save_config_file(api_key)
                print("✅ 設定ファイルに保存しました")
            
            elif choice == '3':
                self._save_plain_api_key(api_key)
                print("✅ プレーンテキストファイルに保存しました")
                print("⚠️ セキュリティ注意: 暗号化ファイルの使用を推奨します")
            
            elif choice == '4':
                print("✅ 今回のみ使用します（保存しません）")
            
            else:
                print("❌ 無効な選択です。今回のみ使用します。")
            
            return api_key
            
        except Exception as e:
            print(f"❌ APIキー設定エラー: {e}")
            exit(1)
    
    def _save_encrypted_api_key(self, api_key: str):
        """APIキーを暗号化して保存"""
        try:
            # 簡易暗号化（base64 + XOR）
            import base64
            key_phrase = input("🔐 暗号化用パスフレーズを入力: ").strip()
            if not key_phrase:
                key_phrase = "default_phrase"
            
            # XOR暗号化
            encrypted_data = ""
            for i, char in enumerate(api_key):
                encrypted_data += chr(ord(char) ^ ord(key_phrase[i % len(key_phrase)]))
            
            # base64エンコード
            encoded_data = base64.b64encode(encrypted_data.encode('utf-8')).decode('utf-8')
            
            encrypted_file = os.path.join(self.dictionary_folder, "api_key.enc")
            with open(encrypted_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'encrypted_key': encoded_data,
                    'created_at': datetime.now().isoformat(),
                    'method': 'xor_base64'
                }, f, indent=2)
            
            print(f"💾 暗号化ファイル保存: {encrypted_file}")
            
        except Exception as e:
            print(f"❌ 暗号化保存エラー: {e}")
            self._save_plain_api_key(api_key)  # フォールバック
    
    def _load_encrypted_api_key(self, encrypted_file: str) -> str:
        """暗号化されたAPIキーを復号"""
        try:
            with open(encrypted_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('method') == 'xor_base64':
                import base64
                key_phrase = input("🔐 暗号化用パスフレーズを入力: ").strip()
                if not key_phrase:
                    key_phrase = "default_phrase"
                
                # base64デコード
                encrypted_data = base64.b64decode(data['encrypted_key']).decode('utf-8')
                
                # XOR復号
                decrypted_key = ""
                for i, char in enumerate(encrypted_data):
                    decrypted_key += chr(ord(char) ^ ord(key_phrase[i % len(key_phrase)]))
                
                return decrypted_key
            
        except Exception as e:
            print(f"❌ 復号エラー: {e}")
            return None
    
    def _save_config_file(self, api_key: str):
        """設定ファイルに保存"""
        config_file = os.path.join(self.dictionary_folder, "config.json")
        config = {}
        
        # 既存設定があれば読み込み
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except:
                config = {}
        
        config.update({
            'claude_api_key': api_key,
            'updated_at': datetime.now().isoformat()
        })
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"💾 設定ファイル保存: {config_file}")
    
    def _save_plain_api_key(self, api_key: str):
        """プレーンテキストで保存（非推奨）"""
        plain_file = os.path.join(self.dictionary_folder, "api_key.txt")
        with open(plain_file, 'w', encoding='utf-8') as f:
            f.write(api_key)
    def validate_youtube_url(self, url: str) -> bool:
        """YouTube URLの妥当性をチェック"""
        if not url or not isinstance(url, str):
            return False
        
        # APIキーっぽい文字列を検出
        if url.startswith(('sk-', 'api-', 'key-')) or len(url) > 100:
            print("⚠️ APIキーのような文字列が入力されています")
            print("💡 YouTube URLを入力してください（例: https://www.youtube.com/watch?v=...）")
            return False
        
        # YouTube URLパターンをチェック
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/v/[\w-]+',
            r'youtube\.com/watch\?v=[\w-]+',
            r'youtu\.be/[\w-]+',
        ]
        
        import re
        for pattern in youtube_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        # より柔軟なチェック（youtube.comまたはyoutu.beを含む）
        if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
            return True
        
        return False
    
    def validate_file_path(self, file_path: str) -> bool:
        """ファイルパスの妥当性をチェック"""
        if not file_path or not isinstance(file_path, str):
            return False
        
        # APIキーっぽい文字列を検出
        if file_path.startswith(('sk-', 'api-', 'key-')) or len(file_path) > 300:
            print("⚠️ APIキーのような文字列が入力されています")
            print("💡 音声ファイルのパスを入力してください（例: C:\\path\\to\\audio.mp3）")
            return False
        
        # ファイル拡張子をチェック
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.avi', '.mkv', '.mov']
        file_lower = file_path.lower()
        
        # 拡張子チェック
        for ext in audio_extensions:
            if file_lower.endswith(ext):
                return True
        
        # パスらしい文字列かチェック
        if ('\\' in file_path or '/' in file_path) and len(file_path) > 3:
            return True
        
        return False
    
    def get_youtube_url_with_validation(self) -> str:
        """検証付きYouTube URL入力"""
        max_attempts = 3
        for attempt in range(max_attempts):
            url = input("🎥 YouTube URL を入力: ").strip()
            
            if not url:
                print("❌ URLが入力されていません")
                continue
            
            if self.validate_youtube_url(url):
                return url
            else:
                print(f"❌ 無効なYouTube URLです ({attempt + 1}/{max_attempts})")
                print("💡 正しい形式: https://www.youtube.com/watch?v=VIDEO_ID")
                if attempt < max_attempts - 1:
                    print("🔄 もう一度入力してください")
        
        print("❌ 有効なYouTube URLが入力されませんでした")
        return None
    
    def get_file_path_with_validation(self) -> str:
        """検証付きファイルパス入力"""
        max_attempts = 3
        for attempt in range(max_attempts):
            file_path = input("🎵 音声ファイルパスを入力: ").strip()
            
            if not file_path:
                print("❌ ファイルパスが入力されていません")
                continue
            
            # 引用符を除去（Windows でファイルをドラッグ&ドロップした場合）
            file_path = file_path.strip('"\'')
            
            if self.validate_file_path(file_path):
                if os.path.exists(file_path):
                    return file_path
                else:
                    print(f"❌ ファイルが存在しません: {file_path}")
                    print("💡 正しいパスを入力するか、ファイルをエクスプローラーからドラッグ&ドロップしてください")
            else:
                print(f"❌ 無効なファイルパスです ({attempt + 1}/{max_attempts})")
                print("💡 サポート形式: .mp3, .wav, .m4a, .flac, .ogg, .mp4, .avi など")
                if attempt < max_attempts - 1:
                    print("🔄 もう一度入力してください")
        
        print("❌ 有効なファイルパスが入力されませんでした")
        return None
    
    def get_folder_path_with_validation(self) -> str:
        """検証付きフォルダパス入力"""
        max_attempts = 3
        for attempt in range(max_attempts):
            folder_path = input("📁 音声ファイルフォルダパスを入力: ").strip()
            
            if not folder_path:
                print("❌ フォルダパスが入力されていません")
                continue
            
            # 引用符を除去
            folder_path = folder_path.strip('"\'')
            
            # APIキーっぽい文字列をチェック
            if folder_path.startswith(('sk-', 'api-', 'key-')) or len(folder_path) > 300:
                print("⚠️ APIキーのような文字列が入力されています")
                print("💡 フォルダのパスを入力してください")
                continue
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                return folder_path
            else:
                print(f"❌ フォルダが存在しません: {folder_path}")
                print("💡 正しいフォルダパスを入力してください")
                if attempt < max_attempts - 1:
                    print("🔄 もう一度入力してください")
        
        print("❌ 有効なフォルダパスが入力されませんでした")
        return None
    
    def load_learned_corrections(self):
        """学習済み修正パターンを読み込み"""
        if os.path.exists(self.learned_corrections_file):
            try:
                with open(self.learned_corrections_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_learned_corrections(self):
        """学習済み修正パターンを保存"""
        with open(self.learned_corrections_file, 'w', encoding='utf-8') as f:
            json.dump(self.learned_corrections, f, ensure_ascii=False, indent=2)
    
    def load_confidence_patterns(self):
        """信頼度パターンを読み込み"""
        if os.path.exists(self.confidence_patterns_file):
            try:
                with open(self.confidence_patterns_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_confidence_patterns(self):
        """信頼度パターンを保存"""
        with open(self.confidence_patterns_file, 'w', encoding='utf-8') as f:
            json.dump(self.confidence_patterns, f, ensure_ascii=False, indent=2)
    
    def learn_from_correction(self, original_text: str, corrected_text: str, low_confidence_words: list):
        """Claude修正結果から学習パターンを抽出"""
        learning_count = 0
        
        # 簡単な単語レベルでの差分検出
        original_words = original_text.split()
        corrected_words = corrected_text.split()
        
        # 新しい修正パターンを検出
        for i, (orig_word, corr_word) in enumerate(zip(original_words, corrected_words)):
            if orig_word != corr_word and len(orig_word) > 2 and len(corr_word) > 2:
                if orig_word not in self.learned_corrections:
                    self.learned_corrections[orig_word] = corr_word
                    learning_count += 1
                    print(f"🧠 学習: '{orig_word}' → '{corr_word}'")
        
        # 低信頼度語のパターン学習
        for word_info in low_confidence_words:
            word = word_info['word'].strip()
            confidence = word_info['confidence']
            
            if word in self.confidence_patterns:
                # 既存パターンの信頼度を更新
                self.confidence_patterns[word]['count'] += 1
                self.confidence_patterns[word]['avg_confidence'] = (
                    self.confidence_patterns[word]['avg_confidence'] + confidence
                ) / 2
            else:
                # 新しい低信頼度パターンを記録
                self.confidence_patterns[word] = {
                    'avg_confidence': confidence,
                    'count': 1,
                    'first_seen': datetime.now().isoformat()
                }
                learning_count += 1
        
        if learning_count > 0:
            self.save_learned_corrections()
            self.save_confidence_patterns()
            self.processing_stats['learned_patterns'] += learning_count
            print(f"📚 {learning_count}個の新しいパターンを学習しました")
    
    def manage_api_key(self):
        """APIキー管理メニュー"""
        print(f"\n🔑 === APIキー管理 ===")
        print("1. 🔄 APIキーを再設定")
        print("2. 📊 現在の設定状況を確認")
        print("3. 🗑️ 保存されたAPIキーを削除")
        print("4. 🔒 暗号化ファイルに変換")
        print("5. ⬅️ メインメニューに戻る")
        
        try:
            choice = input("\n選択してください (1-5): ").strip()
            
            if choice == '1':
                new_key = self._setup_api_key_interactive()
                self.claude_api_key = new_key
                print("✅ APIキーを更新しました")
            
            elif choice == '2':
                self._show_api_key_status()
            
            elif choice == '3':
                self._delete_api_key_files()
            
            elif choice == '4':
                self._convert_to_encrypted()
            
            elif choice == '5':
                return
            
            else:
                print("❌ 無効な選択です")
                
        except Exception as e:
            print(f"❌ エラー: {e}")
    
    def _show_api_key_status(self):
        """APIキー設定状況を表示"""
        print(f"\n📊 === APIキー設定状況 ===")
        
        # 現在使用中のキー
        if self.claude_api_key:
            masked_key = self.claude_api_key[:8] + "*" * (len(self.claude_api_key) - 12) + self.claude_api_key[-4:]
            print(f"🔑 現在のAPIキー: {masked_key}")
        else:
            print("❌ APIキーが設定されていません")
        
        # ファイル存在確認
        files_to_check = [
            ("🔒 暗号化ファイル", os.path.join(self.dictionary_folder, "api_key.enc")),
            ("📄 設定ファイル", os.path.join(self.dictionary_folder, "config.json")),
            ("📝 プレーンファイル", os.path.join(self.dictionary_folder, "api_key.txt"))
        ]
        
        print("\n📁 保存ファイル状況:")
        for name, filepath in files_to_check:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                print(f"  ✅ {name}: 存在 ({size}B, {mtime.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"  ❌ {name}: なし")
        
        # 環境変数確認
        env_key = os.environ.get('CLAUDE_API_KEY')
        if env_key:
            print(f"  ✅ 環境変数: 設定済み")
        else:
            print(f"  ❌ 環境変数: 未設定")
    
    def _delete_api_key_files(self):
        """保存されたAPIキーファイルを削除"""
        files_to_delete = [
            ("🔒 暗号化ファイル", os.path.join(self.dictionary_folder, "api_key.enc")),
            ("📄 設定ファイル", os.path.join(self.dictionary_folder, "config.json")),
            ("📝 プレーンファイル", os.path.join(self.dictionary_folder, "api_key.txt"))
        ]
        
        print(f"\n🗑️ === APIキーファイル削除 ===")
        deleted_count = 0
        
        for name, filepath in files_to_delete:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"  ✅ {name}: 削除完了")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ❌ {name}: 削除失敗 ({e})")
            else:
                print(f"  📭 {name}: ファイルなし")
        
        if deleted_count > 0:
            print(f"\n✅ {deleted_count}個のファイルを削除しました")
            print("⚠️ 次回起動時にAPIキーの再設定が必要です")
        else:
            print("\n📭 削除対象ファイルがありませんでした")
    
    def _convert_to_encrypted(self):
        """既存のプレーンテキストキーを暗号化ファイルに変換"""
        plain_file = os.path.join(self.dictionary_folder, "api_key.txt")
        config_file = os.path.join(self.dictionary_folder, "config.json")
        
        api_key = None
        
        # プレーンファイルから取得
        if os.path.exists(plain_file):
            try:
                with open(plain_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                print("📝 プレーンテキストファイルからAPIキーを取得")
            except Exception as e:
                print(f"❌ プレーンファイル読み込みエラー: {e}")
        
        # 設定ファイルから取得
        elif os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    api_key = config.get('claude_api_key')
                print("📄 設定ファイルからAPIキーを取得")
            except Exception as e:
                print(f"❌ 設定ファイル読み込みエラー: {e}")
        
        if api_key:
            try:
                self._save_encrypted_api_key(api_key)
                print("✅ 暗号化ファイルに変換完了")
                
                # 元ファイルの削除確認
                confirm = input("🗑️ 元のファイルを削除しますか？ (y/N): ").strip().lower()
                if confirm == 'y':
                    if os.path.exists(plain_file):
                        os.remove(plain_file)
                        print("✅ プレーンテキストファイルを削除しました")
            except Exception as e:
                print(f"❌ 暗号化変換エラー: {e}")
        else:
            print("❌ 変換対象のAPIキーが見つかりません")
    
    def _clean_text_basic(self, text: str) -> str:
        """基本的なテキストクリーニング"""
        # メタ情報除去
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text)
        text = re.sub(r'=+', '', text)
        text = re.sub(r':\s*!\s*:', '', text)
        
        # 句読点修正
        text = re.sub(r'。{2,}', '。', text)
        text = re.sub(r'、{2,}', '、', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _apply_whisper_patterns(self, text: str) -> str:
        """Whisper特有の誤字パターン + 学習済みパターンを修正"""
        corrected = text
        
        # 初期辞書パターンを適用
        for error, correction in self.whisper_patterns.items():
            corrected = corrected.replace(error, correction)
        
        # 学習済みパターンを適用
        corrected = self.apply_learned_patterns(corrected)
        
        return corrected
    
    def apply_learned_patterns(self, text: str) -> str:
        """学習済みパターンを適用"""
        corrected = text
        applied_count = 0
        
        # 学習済み修正パターンを適用
        for original, correction in self.learned_corrections.items():
            if original in corrected:
                corrected = corrected.replace(original, correction)
                applied_count += 1
        
        if applied_count > 0:
            print(f"🧠 {applied_count}個の学習済みパターンを適用")
        
        return corrected
    
    def _create_claude_prompt(self, text: str, whisper_confidence_info: str = "") -> str:
        """Claude用の高精度修正プロンプト（口調保持版）"""
        prompt = f"""あなたは日本語の優秀な文字起こし修正エキスパートです。音声認識システム（Whisper）で生成されたテキストを、元の話者の口調や表現を保持しながら、読みやすく自然な日本語に修正してください。

【重要な修正方針】
1. 話者の口調・語尾・感情表現は絶対に変更しない
2. 敬語変換は行わない（「だ・である調」「です・ます調」など元の調子を維持）
3. 方言や独特な表現も保持する
4. 感情的な表現（「〜だろ」「〜だよ」「〜わ」など）も維持

【修正すべき点のみ】
- 明らかな誤字・脱字の修正
- 固有名詞の正確な表記（政治家名、組織名など）
- 適切な句読点の追加（読みやすさのため）
- 音韻的な誤認識の修正（似た音の漢字間違いなど）
- 自然な改行・段落分け

【特に注意する音声認識エラー】
- 政治用語: 「石橋ロシ→石破茂」「地民党→自民党」「急安部派→旧安倍派」
- 組織名: 「党生年局→党執行部」「階段→会談」
- よくある誤字: 「ガソリンを→ガソリン税を」「販売担保→販売価格」「さっさに進化→先走り」

【絶対にやってはいけないこと】
- 口調の変更（関西弁→標準語、カジュアル→丁寧語など）
- 感情表現の削除や変更
- 話者の個性的な表現の修正
- 過度な敬語化

{whisper_confidence_info}

【元テキスト】
{text}

【修正後テキスト】
元の口調と感情を保持しながら、誤字・脱字のみを修正したテキストを出力してください。"""
        
        return prompt
    
    def _call_claude_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Claude APIを呼び出し"""
        headers = {
            "x-api-key": self.claude_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.claude_model,
            "max_tokens": 4000,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.claude_base_url, headers=headers, json=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    return result['content'][0]['text'].strip()
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"⏳ レート制限のため {wait_time} 秒待機中...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Claude API エラー: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"🔄 リクエストエラー (試行 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def _calculate_classification_cost(self, text: str) -> float:
        """分類処理のコストを計算"""
        # 分類用プロンプトは短いので、概算で計算
        input_tokens = (len(text[:500]) + 200) / 1.5  # テキスト + プロンプト
        output_tokens = 10 / 1.5  # カテゴリ名のみ
        input_cost = (input_tokens / 1000000) * 3.00
        output_cost = (output_tokens / 1000000) * 15.00
        return input_cost + output_cost
    
    def _calculate_cost(self, original: str, corrected: str) -> float:
        """テキスト修正のコストを計算"""
        input_tokens = len(original) / 1.5
        output_tokens = len(corrected) / 1.5
        input_cost = (input_tokens / 1000000) * 3.00
        output_cost = (output_tokens / 1000000) * 15.00
        return input_cost + output_cost
    
    def transcribe_with_claude_correction(self, audio_path: str, whisper_model_size: str = "medium") -> Dict:
        """
        Whisper音声認識 + Claude即座修正の統合処理
        """
        start_time = time.time()
        result = {
            'success': False,
            'original_transcript': '',
            'corrected_transcript': '',
            'whisper_info': {},
            'claude_applied': False,
            'cost': 0.0,
            'processing_time': 0.0,
            'confidence_info': '',
            'word_level_data': []
        }
        
        try:
            # Whisper音声認識（詳細情報付き）
            print("🎤 Whisper音声認識実行中...")
            model = whisper.load_model(whisper_model_size)
            
            whisper_result = model.transcribe(
                audio_path,
                language='ja',
                task='transcribe',
                temperature=0.0,
                best_of=5,
                beam_size=5,
                word_timestamps=True,  # 単語レベルタイムスタンプ
                condition_on_previous_text=True
            )
            
            original_text = whisper_result['text']
            result['original_transcript'] = original_text
            result['whisper_info'] = {
                'language': whisper_result.get('language', 'ja'),
                'segments': len(whisper_result.get('segments', [])),
                'duration': sum(seg.get('end', 0) - seg.get('start', 0) 
                             for seg in whisper_result.get('segments', []))
            }
            
            # 信頼度情報の収集
            confidence_scores = []
            word_data = []
            
            for segment in whisper_result.get('segments', []):
                for word_info in segment.get('words', []):
                    if 'probability' in word_info:
                        confidence_scores.append(word_info['probability'])
                        word_data.append({
                            'word': word_info['word'],
                            'confidence': word_info['probability'],
                            'start': word_info.get('start', 0),
                            'end': word_info.get('end', 0)
                        })
            
            result['word_level_data'] = word_data
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                low_confidence_words = [w for w in word_data if w['confidence'] < 0.5]
                
                result['confidence_info'] = f"平均信頼度: {avg_confidence:.3f}, 低信頼度語数: {len(low_confidence_words)}"
                
                # 低信頼度語の情報をClaude用プロンプトに追加
                confidence_info = f"""
【音声認識信頼度情報】
平均信頼度: {avg_confidence:.3f}
低信頼度の語（0.5未満）: {len(low_confidence_words)}個
"""
                if low_confidence_words:
                    confidence_info += "特に注意が必要な語: " + ", ".join([w['word'] for w in low_confidence_words[:10]])
            else:
                confidence_info = ""
            
            print(f"✅ Whisper認識完了 (信頼度: {result['confidence_info']})")
            
            # 基本的な前処理
            cleaned_text = self._clean_text_basic(original_text)
            pattern_corrected = self._apply_whisper_patterns(cleaned_text)
            
            # Claude APIによる高精度修正
            print("🤖 Claude 3.5 Sonnetによる高精度修正中...")
            claude_prompt = self._create_claude_prompt(pattern_corrected, confidence_info)
            claude_corrected = self._call_claude_api(claude_prompt)
            
            if claude_corrected:
                result['corrected_transcript'] = claude_corrected
                result['claude_applied'] = True
                result['cost'] = self._calculate_cost(original_text, claude_corrected)
                print(f"✅ Claude修正完了 (コスト: ${result['cost']:.4f})")
                
                # 修正結果から学習
                self.learn_from_correction(
                    original_text, 
                    claude_corrected, 
                    result['word_level_data']
                )
                
                self.processing_stats['claude_corrections'] += 1
                self.processing_stats['total_cost'] += result['cost']
            else:
                result['corrected_transcript'] = pattern_corrected
                print("⚠️ Claude修正失敗、基本修正のみ適用")
            
            result['success'] = True
            self.processing_stats['successful_transcriptions'] += 1
            
        except Exception as e:
            print(f"❌ 音声処理エラー: {e}")
            result['error'] = str(e)
        
        finally:
            result['processing_time'] = time.time() - start_time
            self.processing_stats['processing_time'] += result['processing_time']
        
        return result

class IntegratedVoiceRAGSystem:
    """統合音声RAGシステム（Claude修正機能付き）"""
    
    def __init__(self, db_path="voice_rag_db.json", output_folder="D:\\Python\\テキスト修正"):
        print("🚀 統合システム初期化中...")
        
        self.db_path = db_path
        self.output_folder = output_folder
        self.claude_integration = ClaudeWhisperIntegration()
        self.sentence_model = None
        self.database = self.load_database()
        
        # カテゴリとフォルダの対応
        self.category_folders = {
            'ビジネス・経営': 'business', '技術・プログラミング': 'technology', 
            '教育・学習': 'education', 'エンターテイメント': 'entertainment',
            '日常・ライフスタイル': 'lifestyle', 'ニュース・時事': 'news',
            '健康・医療': 'health', '料理・グルメ': 'cooking',
            '旅行・観光': 'travel', 'スポーツ・フィットネス': 'sports',
            '音楽・アート': 'music_art', 'ゲーム': 'gaming',
            '科学・研究': 'science', '政治・社会': 'politics', 'その他': 'others'
        }
        
        # カテゴリ分類用キーワード（フォールバック用）
        self.category_keywords = {
            'ビジネス・経営': ['会議', '売上', 'マーケティング', '戦略', '予算', '顧客', '営業', '経営', '投資', 'ビジネス', '企業', '会社'],
            '技術・プログラミング': ['プログラミング', 'AI', 'システム', 'アプリ', 'ソフトウェア', 'データベース', 'API', 'クラウド', 'アルゴリズム', 'コンピュータ', 'IT', 'DX'],
            '政治・社会': ['政治', '選挙', '政策', '国会', '議員', '政府', '自民党', '政党', '首相', '大臣', '法律', '制度', '改革', '社会問題'],
            'ニュース・時事': ['ニュース', '報道', '事件', '社会', '経済', '国際', '速報', '記者会見', '発表', '調査'],
            'エンターテイメント': ['映画', '音楽', 'アニメ', 'ゲーム', '芸能', 'ちいかわ', 'ドラマ', 'バラエティ', 'アイドル', '俳優', '歌手'],
            '料理・グルメ': ['料理', 'レシピ', '食材', 'グルメ', 'あげ玉', 'そうめん', '食事', '調理', 'レストラン', '美味しい', '食べ物'],
            'スポーツ・フィットネス': ['スポーツ', 'フィットネス', '運動', 'トレーニング', '競技', '筋トレ', '野球', 'サッカー', 'MVP', '大谷翔平', '大谷', 'ジャッジ', 'ドジャース', 'ヤンキース', 'メジャーリーグ', 'MLB', 'バスケ', 'テニス', 'ゴルフ'],
            '教育・学習': ['学習', '授業', '講義', '試験', '研究', '論文', '学生', '教育', '勉強', '資格', '大学', '学校'],
            '健康・医療': ['健康', '医療', '病気', '治療', 'ダイエット', 'フィットネス', '栄養', '予防', '薬', '医師'],
            '日常・ライフスタイル': ['家族', '友人', '買い物', '生活', '趣味', 'ファッション', '美容', 'インテリア', '日常', 'ライフスタイル'],
            '旅行・観光': ['旅行', '観光', 'ホテル', '観光地', '海外', '国内旅行', '温泉', 'リゾート'],
            '音楽・アート': ['音楽', 'アート', '芸術', '楽器', '絵画', 'デザイン', 'クリエイティブ'],
            'ゲーム': ['ビデオゲーム', 'ゲーム実況', 'eスポーツ', 'ゲーム開発', 'プレイ'],
            '科学・研究': ['科学', '研究', '実験', '理論', '発見', '学術論文', '技術革新']
        }
        
        self.create_category_folders()
        print("✅ 統合システム初期化完了")
    
    def validate_youtube_url(self, url: str) -> bool:
        """YouTube URLの妥当性をチェック"""
        if not url or not isinstance(url, str):
            return False
        
        # APIキーっぽい文字列を検出
        if url.startswith(('sk-', 'api-', 'key-')) or len(url) > 100:
            print("⚠️ APIキーのような文字列が入力されています")
            print("💡 YouTube URLを入力してください（例: https://www.youtube.com/watch?v=...）")
            return False
        
        # YouTube URLパターンをチェック
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/v/[\w-]+',
            r'youtube\.com/watch\?v=[\w-]+',
            r'youtu\.be/[\w-]+',
        ]
        
        import re
        for pattern in youtube_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        # より柔軟なチェック（youtube.comまたはyoutu.beを含む）
        if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
            return True
        
        return False
    
    def validate_file_path(self, file_path: str) -> bool:
        """ファイルパスの妥当性をチェック"""
        if not file_path or not isinstance(file_path, str):
            return False
        
        # APIキーっぽい文字列を検出
        if file_path.startswith(('sk-', 'api-', 'key-')) or len(file_path) > 300:
            print("⚠️ APIキーのような文字列が入力されています")
            print("💡 音声ファイルのパスを入力してください（例: C:\\path\\to\\audio.mp3）")
            return False
        
        # ファイル拡張子をチェック
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.avi', '.mkv', '.mov']
        file_lower = file_path.lower()
        
        # 拡張子チェック
        for ext in audio_extensions:
            if file_lower.endswith(ext):
                return True
        
        # パスらしい文字列かチェック
        if ('\\' in file_path or '/' in file_path) and len(file_path) > 3:
            return True
        
        return False
    
    def get_youtube_url_with_validation(self) -> str:
        """検証付きYouTube URL入力"""
        max_attempts = 3
        for attempt in range(max_attempts):
            url = input("🎥 YouTube URL を入力: ").strip()
            
            if not url:
                print("❌ URLが入力されていません")
                continue
            
            if self.validate_youtube_url(url):
                return url
            else:
                print(f"❌ 無効なYouTube URLです ({attempt + 1}/{max_attempts})")
                print("💡 正しい形式: https://www.youtube.com/watch?v=VIDEO_ID")
                if attempt < max_attempts - 1:
                    print("🔄 もう一度入力してください")
        
        print("❌ 有効なYouTube URLが入力されませんでした")
        return None
    
    def get_file_path_with_validation(self) -> str:
        """検証付きファイルパス入力"""
        max_attempts = 3
        for attempt in range(max_attempts):
            file_path = input("🎵 音声ファイルパスを入力: ").strip()
            
            if not file_path:
                print("❌ ファイルパスが入力されていません")
                continue
            
            # 引用符を除去（Windows でファイルをドラッグ&ドロップした場合）
            file_path = file_path.strip('"\'')
            
            if self.validate_file_path(file_path):
                if os.path.exists(file_path):
                    return file_path
                else:
                    print(f"❌ ファイルが存在しません: {file_path}")
                    print("💡 正しいパスを入力するか、ファイルをエクスプローラーからドラッグ&ドロップしてください")
            else:
                print(f"❌ 無効なファイルパスです ({attempt + 1}/{max_attempts})")
                print("💡 サポート形式: .mp3, .wav, .m4a, .flac, .ogg, .mp4, .avi など")
                if attempt < max_attempts - 1:
                    print("🔄 もう一度入力してください")
        
        print("❌ 有効なファイルパスが入力されませんでした")
        return None
    
    def get_folder_path_with_validation(self) -> str:
        """検証付きフォルダパス入力"""
        max_attempts = 3
        for attempt in range(max_attempts):
            folder_path = input("📁 音声ファイルフォルダパスを入力: ").strip()
            
            if not folder_path:
                print("❌ フォルダパスが入力されていません")
                continue
            
            # 引用符を除去
            folder_path = folder_path.strip('"\'')
            
            # APIキーっぽい文字列をチェック
            if folder_path.startswith(('sk-', 'api-', 'key-')) or len(folder_path) > 300:
                print("⚠️ APIキーのような文字列が入力されています")
                print("💡 フォルダのパスを入力してください")
                continue
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                return folder_path
            else:
                print(f"❌ フォルダが存在しません: {folder_path}")
                print("💡 正しいフォルダパスを入力してください")
                if attempt < max_attempts - 1:
                    print("🔄 もう一度入力してください")
        
        print("❌ 有効なフォルダパスが入力されませんでした")
        return None
    
    def show_help_and_examples(self):
        """ヘルプとサンプル表示"""
        print(f"\n📖 === 使用方法とサンプル ===")
        print("🎥 YouTube URL の例:")
        print("  ✅ https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  ✅ https://youtu.be/dQw4w9WgXcQ")
        print("  ✅ youtube.com/watch?v=dQw4w9WgXcQ")
        
        print("\n🎵 音声ファイルパスの例:")
        print("  ✅ C:\\Users\\Username\\Music\\audio.mp3")
        print("  ✅ D:\\recordings\\meeting.wav")
        print("  ✅ ./audio/sample.m4a")
        
        print("\n📁 フォルダパスの例:")
        print("  ✅ C:\\Users\\Username\\Music\\")
        print("  ✅ D:\\recordings\\")
        
        print("\n💡 入力のコツ:")
        print("  • ファイル/フォルダはエクスプローラーからドラッグ&ドロップ可能")
        print("  • パスに空白がある場合は自動で引用符を処理")
        print("  • サポート音声形式: mp3, wav, m4a, flac, ogg, mp4, avi")
        
        print("\n🔑 APIキー管理:")
        print("  • 初回起動時に自動設定")
        print("  • 暗号化保存を推奨")
        print("  • 環境変数での管理も可能")
        
        print("\n🧠 学習機能:")
        print("  • Claude修正結果から自動学習")
        print("  • 低信頼度語パターンを蓄積")
        print("  • 次回処理時に学習内容を自動適用")
    
    def create_category_folders(self):
        """カテゴリ別フォルダを作成"""
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            print(f"📁 メイン出力フォルダ確認: {self.output_folder}")
            
            for category, folder_name in self.category_folders.items():
                folder_path = os.path.join(self.output_folder, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                print(f"   📂 {category} → {folder_path}")
                
        except Exception as e:
            print(f"❌ フォルダ作成エラー: {e}")
            print(f"❌ 問題のパス: {self.output_folder}")
    
    def classify_text(self, text):
        """Claude APIによる高精度カテゴリ分類"""
        print("🎯 Claude APIでカテゴリ分類中...")
        
        # まず基本的なキーワード分類を試行（フォールバック用）
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword.lower())
            category_scores[category] = score
        
        fallback_category = max(category_scores, key=category_scores.get) if max(category_scores.values()) > 0 else "その他"
        
        # Claude APIによる高精度分類
        claude_category = self._classify_with_claude_api(text)
        if claude_category:
            print(f"🤖 Claude分類: {claude_category}")
            print(f"📝 キーワード分類: {fallback_category}")
            return claude_category
        else:
            print(f"⚠️ Claude分類失敗、キーワード分類使用: {fallback_category}")
            return fallback_category
    
    def _classify_with_claude_api(self, text: str) -> Optional[str]:
        """Claude APIによるカテゴリ分類"""
        categories_list = "\n".join([f"- {cat}" for cat in self.category_folders.keys()])
        
        prompt = f"""以下のテキストを最適なカテゴリに分類してください。

【利用可能カテゴリ】
{categories_list}

【分類ルール】
- テキストの主な内容に基づいて分類
- スポーツ関連（野球、サッカー、MVPなど）は「スポーツ・フィットネス」
- 政治家、政党、選挙関連は「政治・社会」
- 技術、プログラミング、AI関連は「技術・プログラミング」
- 料理、食材、レシピ関連は「料理・グルメ」
- エンタメ、芸能、キャラクター関連は「エンターテイメント」
- どのカテゴリにも当てはまらない場合は「その他」

【テキスト】
{text[:500]}...

【出力形式】
カテゴリ名のみを出力してください（説明不要）。上記のカテゴリリストから正確に選択してください。"""
        
        try:
            headers = {
                "x-api-key": self.claude_integration.claude_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.claude_integration.claude_model,
                "max_tokens": 50,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(self.claude_integration.claude_base_url, 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                category = result['content'][0]['text'].strip()
                
                if category in self.category_folders:
                    # 分類コストを記録
                    classification_cost = self._classify_with_claude_api_cost_calc(text)
                    self.claude_integration.processing_stats['classification_cost'] += classification_cost
                    self.claude_integration.processing_stats['claude_classifications'] += 1
                    return category
                else:
                    print(f"⚠️ 無効なカテゴリ: {category}")
                    return None
            else:
                print(f"❌ 分類API エラー: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ 分類API エラー: {e}")
            return None
    
    def _classify_with_claude_api_cost_calc(self, text: str) -> float:
        """分類API のコスト計算"""
        return self.claude_integration._calculate_classification_cost(text)
    
    def load_sentence_model(self):
        """文埋め込みモデルの読み込み"""
        if self.sentence_model is None:
            print("📚 文埋め込みモデル読み込み中...")
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ 文埋め込みモデル読み込み完了")
    
    def create_embedding(self, text):
        """テキストの埋め込みベクトル作成"""
        if self.sentence_model is None:
            self.load_sentence_model()
        return self.sentence_model.encode([text])[0].tolist()
    
    def save_transcript_to_file(self, title, original_transcript, corrected_transcript, 
                              category, source_path, processing_info):
        """高品質な文字起こしファイルを保存"""
        try:
            folder_name = self.category_folders.get(category, 'others')
            category_folder = os.path.join(self.output_folder, folder_name)
            
            print(f"📁 保存先フォルダ: {category_folder}")
            
            # フォルダ存在確認と作成
            os.makedirs(category_folder, exist_ok=True)
            if not os.path.exists(category_folder):
                raise Exception(f"フォルダ作成失敗: {category_folder}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_title = re.sub(r'[^\w\s-]', '_', title)[:30]
            filename = f"{timestamp}_{safe_title}.txt"
            file_path = os.path.join(category_folder, filename)
            
            print(f"📄 保存ファイル名: {filename}")
            print(f"📍 完全パス: {file_path}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"タイトル: {title}\n")
                f.write(f"カテゴリ: {category}\n")
                f.write(f"ソース: {source_path}\n")
                f.write(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"処理時間: {processing_info.get('processing_time', 0):.2f}秒\n")
                f.write(f"Claude修正: {'✅ 適用' if processing_info.get('claude_applied') else '❌ 未適用'}\n")
                f.write(f"処理コスト: ${processing_info.get('cost', 0):.4f}\n")
                f.write(f"Whisper信頼度: {processing_info.get('confidence_info', 'N/A')}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("【📝 最終修正版テキスト】\n")
                f.write("-" * 60 + "\n")
                f.write(corrected_transcript)
                f.write("\n" + "-" * 60 + "\n\n")
                
                f.write("【🎤 Whisper原文】\n")
                f.write("-" * 40 + "\n")
                f.write(original_transcript)
                f.write("\n" + "-" * 40 + "\n\n")
                
                f.write("【📊 処理詳細】\n")
                f.write(f"原文文字数: {len(original_transcript)}\n")
                f.write(f"修正後文字数: {len(corrected_transcript)}\n")
                f.write(f"文字数変化: {len(corrected_transcript) - len(original_transcript):+d}\n")
                
                # 低信頼度語の詳細
                word_data = processing_info.get('word_level_data', [])
                if word_data:
                    low_conf_words = [w for w in word_data if w.get('confidence', 1) < 0.5]
                    if low_conf_words:
                        f.write(f"\n【⚠️ 低信頼度語 ({len(low_conf_words)}個)】\n")
                        for word in low_conf_words[:10]:
                            f.write(f"  '{word['word']}' (信頼度: {word['confidence']:.3f})\n")
            
            # ファイル存在確認
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"💾 ✅ ファイル保存成功: {file_path}")
                print(f"📏 ファイルサイズ: {file_size} bytes")
                return file_path
            else:
                raise Exception("ファイル保存後に存在確認失敗")
            
        except Exception as e:
            print(f"❌ ファイル保存エラー: {e}")
            print(f"❌ 対象フォルダ: {category_folder}")
            print(f"❌ 対象ファイル名: {filename}")
            import traceback
            traceback.print_exc()
            return None
    
    def add_to_database(self, source_type, source_path, title, original_transcript, 
                       corrected_transcript, category, processing_info):
        """データベースに追加"""
        embedding = self.create_embedding(corrected_transcript)
        
        entry = {
            'id': len(self.database) + 1,
            'source_type': source_type,
            'source_path': source_path,
            'title': title,
            'original_transcript': original_transcript,
            'corrected_transcript': corrected_transcript,
            'category': category,
            'embedding': embedding,
            'created_at': datetime.now().isoformat(),
            'processing_info': processing_info,
            'word_count': len(corrected_transcript.split()),
            'char_count': len(corrected_transcript)
        }
        
        self.database.append(entry)
        self.save_database()
        
        # ファイル保存
        saved_file = self.save_transcript_to_file(
            title, original_transcript, corrected_transcript, 
            category, source_path, processing_info
        )
        
        if saved_file:
            entry['saved_file_path'] = saved_file
            self.save_database()
        
        print(f"✅ データベース追加完了: {title}")
        print(f"📂 カテゴリ: {category}")
        print(f"💰 コスト: ${processing_info.get('cost', 0):.4f}")
    
    def download_youtube_audio(self, url, output_path=None):
        """YouTube音声ダウンロード"""
        if output_path is None:
            output_path = tempfile.mkdtemp()
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'unknown')
                
                for file in os.listdir(output_path):
                    if file.endswith('.mp3'):
                        return os.path.join(output_path, file), title
                        
        except Exception as e:
            print(f"❌ YouTube音声取得エラー: {e}")
            return None, None
    
    def process_youtube_url(self, url, whisper_model_size="medium"):
        """YouTube URL処理（統合版）"""
        print(f"🎥 YouTube URL処理開始: {url}")
        
        audio_path, title = self.download_youtube_audio(url)
        if not audio_path:
            print("❌ YouTube音声取得失敗")
            return False
        
        try:
            # Whisper + Claude統合処理
            result = self.claude_integration.transcribe_with_claude_correction(
                audio_path, whisper_model_size
            )
            
            if not result['success']:
                print("❌ 音声処理失敗")
                return False
            
            # カテゴリ分類（Claude API使用）
            category = self.classify_text(result['corrected_transcript'])
            
            # データベース追加
            self.add_to_database(
                'youtube', url, title,
                result['original_transcript'],
                result['corrected_transcript'],
                category, result
            )
            
            print(f"🎉 YouTube処理完了!")
            print(f"📝 原文: {len(result['original_transcript'])}文字")
            print(f"✨ 修正版: {len(result['corrected_transcript'])}文字")
            print(f"🤖 Claude修正: {'✅' if result['claude_applied'] else '❌'}")
            
            return True
            
        finally:
            try:
                os.remove(audio_path)
            except:
                pass
    
    def process_audio_file(self, file_path, whisper_model_size="medium"):
        """音声ファイル処理（統合版）"""
        if not os.path.exists(file_path):
            print(f"❌ ファイル不存在: {file_path}")
            return False
        
        print(f"🎵 音声ファイル処理開始: {os.path.basename(file_path)}")
        
        # Whisper + Claude統合処理
        result = self.claude_integration.transcribe_with_claude_correction(
            file_path, whisper_model_size
        )
        
        if not result['success']:
            print("❌ 音声処理失敗")
            return False
        
        # カテゴリ分類（Claude API使用）
        category = self.classify_text(result['corrected_transcript'])
        title = os.path.basename(file_path)
        
        # データベース追加
        self.add_to_database(
            'file', file_path, title,
            result['original_transcript'],
            result['corrected_transcript'],
            category, result
        )
        
        print(f"🎉 ファイル処理完了!")
        print(f"📝 原文: {len(result['original_transcript'])}文字")
        print(f"✨ 修正版: {len(result['corrected_transcript'])}文字")
        print(f"🤖 Claude修正: {'✅' if result['claude_applied'] else '❌'}")
        
        return True
    
    def search_similar(self, query, top_k=3):
        """類似検索"""
        if not self.database:
            return []
        
        if self.sentence_model is None:
            self.load_sentence_model()
        
        query_embedding = self.sentence_model.encode([query])[0]
        similarities = []
        
        for entry in self.database:
            entry_embedding = np.array(entry['embedding'])
            similarity = cosine_similarity([query_embedding], [entry_embedding])[0][0]
            similarities.append((entry, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def search_and_display(self, query):
        """検索結果表示"""
        print(f"\n🔍 検索クエリ: '{query}'")
        print("-" * 50)
        
        results = self.search_similar(query)
        
        if not results:
            print("❌ 検索結果なし")
            return
        
        for i, (entry, similarity) in enumerate(results, 1):
            print(f"\n{i}. 📄 {entry['title']}")
            print(f"   📂 カテゴリ: {entry['category']}")
            print(f"   🎯 類似度: {similarity:.3f}")
            print(f"   🎤 Whisper原文字数: {len(entry['original_transcript'])}")
            print(f"   ✨ Claude修正後: {len(entry['corrected_transcript'])}")
            print(f"   🤖 Claude適用: {'✅' if entry.get('processing_info', {}).get('claude_applied') else '❌'}")
            print(f"   💰 処理コスト: ${entry.get('processing_info', {}).get('cost', 0):.4f}")
            print(f"   📅 作成日: {entry['created_at'][:10]}")
            print(f"   📝 内容: {entry['corrected_transcript'][:100]}...")
    
    def show_processing_stats(self):
        """処理統計表示"""
        stats = self.claude_integration.processing_stats
        print(f"\n📊 === 処理統計 ===")
        print(f"📊 総処理ファイル数: {stats['total_files']}")
        print(f"✅ 成功した音声認識: {stats['successful_transcriptions']}")
        print(f"🤖 Claude修正適用: {stats['claude_corrections']}")
        print(f"🎯 Claude分類適用: {stats['claude_classifications']}")
        print(f"💰 総処理コスト: ${stats['total_cost']:.4f}")
        print(f"💰 分類コスト: ${stats['classification_cost']:.4f}")
        print(f"⏱️ 総処理時間: {stats['processing_time']:.1f}秒")
        print(f"🧠 学習済みパターン数: {len(self.claude_integration.learned_corrections)}")
        print(f"📚 新規学習パターン: {stats['learned_patterns']}")
        
        if stats['successful_transcriptions'] > 0:
            print(f"📈 平均処理時間: {stats['processing_time']/stats['successful_transcriptions']:.1f}秒/ファイル")
            print(f"💵 平均修正コスト: ${stats['total_cost']/stats['successful_transcriptions']:.4f}/ファイル")
            if stats['claude_classifications'] > 0:
                print(f"🎯 平均分類コスト: ${stats['classification_cost']/stats['claude_classifications']:.4f}/ファイル")
            print(f"📊 Claude分類精度: {stats['claude_classifications']}/{stats['successful_transcriptions']} ({stats['claude_classifications']/stats['successful_transcriptions']*100:.1f}%)")
    
    def show_learned_patterns(self):
        """学習済みパターンを表示"""
        print(f"\n🧠 === 学習済み修正パターン ===")
        if not self.claude_integration.learned_corrections:
            print("📭 学習済みパターンはありません")
            return
        
        print(f"📚 総パターン数: {len(self.claude_integration.learned_corrections)}")
        print("=" * 60)
        
        for i, (original, corrected) in enumerate(self.claude_integration.learned_corrections.items(), 1):
            print(f"{i:2d}. '{original}' → '{corrected}'")
        
        print("\n🎯 === 低信頼度語パターン ===")
        if not self.claude_integration.confidence_patterns:
            print("📭 低信頼度語パターンはありません")
            return
        
        print(f"📊 総パターン数: {len(self.claude_integration.confidence_patterns)}")
        print("=" * 60)
        
        # 信頼度の低い順にソート
        sorted_patterns = sorted(
            self.claude_integration.confidence_patterns.items(),
            key=lambda x: x[1]['avg_confidence']
        )
        
        for i, (word, info) in enumerate(sorted_patterns[:20], 1):  # 上位20件
            print(f"{i:2d}. '{word}' (信頼度: {info['avg_confidence']:.3f}, 出現: {info['count']}回)")
    
    def clear_learned_patterns(self):
        """学習済みパターンをクリア"""
        confirm = input("🗑️ 学習済みパターンを全てクリアしますか？ (y/N): ").strip().lower()
        if confirm == 'y':
            self.claude_integration.learned_corrections = {}
            self.claude_integration.confidence_patterns = {}
            self.claude_integration.save_learned_corrections()
            self.claude_integration.save_confidence_patterns()
            print("✅ 学習済みパターンをクリアしました")
        else:
            print("❌ キャンセルしました")
    def show_help_and_examples(self):
        """ヘルプとサンプル表示"""
        print(f"\n📖 === 使用方法とサンプル ===")
        print("🎥 YouTube URL の例:")
        print("  ✅ https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  ✅ https://youtu.be/dQw4w9WgXcQ")
        print("  ✅ youtube.com/watch?v=dQw4w9WgXcQ")
        
        print("\n🎵 音声ファイルパスの例:")
        print("  ✅ C:\\Users\\Username\\Music\\audio.mp3")
        print("  ✅ D:\\recordings\\meeting.wav")
        print("  ✅ ./audio/sample.m4a")
        
        print("\n📁 フォルダパスの例:")
        print("  ✅ C:\\Users\\Username\\Music\\")
        print("  ✅ D:\\recordings\\")
        
        print("\n💡 入力のコツ:")
        print("  • ファイル/フォルダはエクスプローラーからドラッグ&ドロップ可能")
        print("  • パスに空白がある場合は自動で引用符を処理")
        print("  • サポート音声形式: mp3, wav, m4a, flac, ogg, mp4, avi")
        
        print("\n🔑 APIキー管理:")
        print("  • 初回起動時に自動設定")
        print("  • 暗号化保存を推奨")
        print("  • 環境変数での管理も可能")
        
        print("\n🧠 学習機能:")
        print("  • Claude修正結果から自動学習")
        print("  • 低信頼度語パターンを蓄積")
        print("  • 次回処理時に学習内容を自動適用")
    
    def load_database(self):
        """データベース読み込み"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_database(self):
        """データベース保存"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, ensure_ascii=False, indent=2)
    
    def open_category_folder(self):
        """カテゴリフォルダを開く"""
        print(f"\n📁 === カテゴリ別フォルダ ===")
        print(f"📂 メインフォルダ: {self.output_folder}")
        
        categories = list(self.category_folders.keys())
        for i, category in enumerate(categories, 1):
            folder_name = self.category_folders[category]
            folder_path = os.path.join(self.output_folder, folder_name)
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.txt')]) if os.path.exists(folder_path) else 0
            print(f"{i:2d}. {category} ({folder_name}) - {file_count}件")
        
        print(f"{len(categories)+1:2d}. メインフォルダを開く")
        
        try:
            choice = int(input(f"\n開きたいフォルダ番号 (1-{len(categories)+1}): "))
            if 1 <= choice <= len(categories):
                category = categories[choice-1]
                folder_name = self.category_folders[category]
                folder_path = os.path.join(self.output_folder, folder_name)
                os.startfile(folder_path)
                print(f"✅ フォルダを開きました: {folder_path}")
            elif choice == len(categories)+1:
                os.startfile(self.output_folder)
                print(f"✅ メインフォルダを開きました: {self.output_folder}")
            else:
                print("❌ 無効な番号です")
        except ValueError:
            print("❌ 数字を入力してください")
        except Exception as e:
            print(f"❌ フォルダを開けませんでした: {e}")
    
    def show_database_stats(self):
        """データベース統計表示"""
        if not self.database:
            print("📭 データベースは空です")
            return
        
        print(f"\n📊 === データベース統計 ===")
        print(f"📄 総エントリ数: {len(self.database)}")
        
        categories = {}
        claude_applied = 0
        total_cost = 0.0
        total_words = 0
        
        for entry in self.database:
            category = entry['category']
            categories[category] = categories.get(category, 0) + 1
            total_words += entry['word_count']
            
            processing_info = entry.get('processing_info', {})
            if processing_info.get('claude_applied'):
                claude_applied += 1
            total_cost += processing_info.get('cost', 0)
        
        print(f"📝 総単語数: {total_words:,}")
        print(f"📊 平均単語数: {total_words//len(self.database):,}")
        print(f"🤖 Claude修正適用: {claude_applied}件 ({claude_applied/len(self.database)*100:.1f}%)")
        print(f"💰 総処理コスト: ${total_cost:.4f}")
        
        print("\n📂 カテゴリ別分布:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}件")
    
    def show_transcript_details(self):
        """文字起こし詳細表示"""
        if not self.database:
            print("📭 データベースは空です")
            return
        
        print(f"\n📋 === 保存されている文字起こし一覧 ===")
        for i, entry in enumerate(self.database, 1):
            processing_info = entry.get('processing_info', {})
            claude_status = "🤖✅" if processing_info.get('claude_applied') else "❌"
            
            print(f"\n{i:2d}. 📄 {entry['title']}")
            print(f"    📂 カテゴリ: {entry['category']}")
            print(f"    🎤 ソース: {entry['source_type']}")
            print(f"    {claude_status} Claude修正: {'適用' if processing_info.get('claude_applied') else '未適用'}")
            print(f"    💰 コスト: ${processing_info.get('cost', 0):.4f}")
            print(f"    📅 作成日: {entry['created_at'][:10]}")
            print(f"    📝 文字数: 原文{len(entry.get('original_transcript', ''))} → 修正後{len(entry['corrected_transcript'])}")
        
        try:
            choice = int(input(f"\n詳細を見たい項目番号 (1-{len(self.database)}): "))
            if 1 <= choice <= len(self.database):
                entry = self.database[choice - 1]
                processing_info = entry.get('processing_info', {})
                
                print(f"\n" + "="*80)
                print(f"📄 タイトル: {entry['title']}")
                print(f"📂 カテゴリ: {entry['category']}")
                print(f"🎤 ソース: {entry['source_path']}")
                print(f"📅 作成日時: {entry['created_at']}")
                print(f"⏱️ 処理時間: {processing_info.get('processing_time', 0):.2f}秒")
                print(f"🤖 Claude修正: {'✅ 適用' if processing_info.get('claude_applied') else '❌ 未適用'}")
                print(f"💰 処理コスト: ${processing_info.get('cost', 0):.4f}")
                print(f"🎯 信頼度情報: {processing_info.get('confidence_info', 'N/A')}")
                
                print(f"\n📊 統計:")
                print(f"  原文文字数: {len(entry.get('original_transcript', ''))}文字")
                print(f"  修正後文字数: {len(entry['corrected_transcript'])}文字")
                print(f"  単語数: {entry['word_count']}語")
                
                print(f"\n【✨ Claude修正版テキスト】")
                print("-" * 80)
                print(entry['corrected_transcript'])
                
                if entry.get('original_transcript'):
                    print(f"\n【🎤 Whisper原文】")
                    print("-" * 60)
                    print(entry['original_transcript'])
                
                print("-" * 80)
            else:
                print("❌ 無効な番号です")
        except ValueError:
            print("❌ 数字を入力してください")
    
    def batch_process_folder(self, folder_path, whisper_model_size="medium"):
        """フォルダ内音声ファイル一括処理"""
        if not os.path.exists(folder_path):
            print(f"❌ フォルダが存在しません: {folder_path}")
            return
        
        # サポート音声形式
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.avi']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(folder_path).glob(f"*{ext}"))
            audio_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not audio_files:
            print(f"❌ 音声ファイルが見つかりません: {folder_path}")
            return
        
        print(f"🎵 {len(audio_files)}個の音声ファイルを発見")
        print(f"📁 処理フォルダ: {folder_path}")
        print(f"🤖 Whisperモデル: {whisper_model_size}")
        
        confirm = input(f"\n一括処理を開始しますか？ (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 処理をキャンセルしました")
            return
        
        print(f"\n🚀 一括処理開始 ({len(audio_files)}ファイル)")
        print("="*80)
        
        success_count = 0
        total_cost = 0.0
        start_time = time.time()
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] 📁 {audio_file.name}")
            
            if self.process_audio_file(str(audio_file), whisper_model_size):
                success_count += 1
                # 最後の処理のコスト情報を取得
                if self.database:
                    last_entry = self.database[-1]
                    cost = last_entry.get('processing_info', {}).get('cost', 0)
                    total_cost += cost
                
                print(f"    ✅ 成功 (累計コスト: ${total_cost:.4f})")
            else:
                print(f"    ❌ 失敗")
            
            # 進捗表示
            progress = (i / len(audio_files)) * 100
            print(f"    📊 進捗: {progress:.1f}% ({success_count}/{i}成功)")
            
            # APIレート制限対策
            time.sleep(1)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 === 一括処理完了 ===")
        print(f"📊 結果: {success_count}/{len(audio_files)}ファイル成功")
        print(f"⏱️ 総処理時間: {total_time:.1f}秒")
        print(f"💰 総コスト: ${total_cost:.4f}")
        if success_count > 0:
            print(f"📈 平均処理時間: {total_time/success_count:.1f}秒/ファイル")
            print(f"💵 平均コスト: ${total_cost/success_count:.4f}/ファイル")


def main():
    """メイン実行関数"""
    print("🎵" * 20)
    print("🎤 Whisper + Claude 3.5 Sonnet 統合音声処理システム")
    print("🎵" * 20)
    print("✨ 高精度音声認識 + AI文章修正の最強組み合わせ！")
    
    try:
        rag = IntegratedVoiceRAGSystem()
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        return
    
    while True:
        print("\n" + "="*60)
        print("🎛️  メニュー")
        print("="*60)
        print("1. 🎥 YouTube URLから音声を処理")
        print("2. 🎵 音声ファイルを処理")
        print("3. 📁 フォルダ内音声ファイル一括処理")
        print("4. 🔍 テキスト検索")
        print("5. 📊 データベース統計表示")
        print("6. 📋 文字起こし内容の詳細表示")
        print("7. 📂 カテゴリ別フォルダを開く")
        print("8. 📈 処理統計表示")
        print("9. 🧠 学習済みパターン表示")
        print("10. 🗑️ 学習済みパターンクリア")
        print("11. 🔑 APIキー管理")
        print("12. 📖 ヘルプ・使用方法")
        print("13. ❌ 終了")
        
        try:
            choice = input("\n選択してください (1-13): ").strip()
            
            if choice == '1':
                url = rag.get_youtube_url_with_validation()
                if url:
                    model_size = input("Whisperモデルサイズ (small/medium/large) [medium]: ").strip() or "medium"
                    rag.process_youtube_url(url, model_size)
            
            elif choice == '2':
                file_path = rag.get_file_path_with_validation()
                if file_path:
                    model_size = input("Whisperモデルサイズ (small/medium/large) [medium]: ").strip() or "medium"
                    rag.process_audio_file(file_path, model_size)
            
            elif choice == '3':
                folder_path = rag.get_folder_path_with_validation()
                if folder_path:
                    model_size = input("Whisperモデルサイズ (small/medium/large) [medium]: ").strip() or "medium"
                    rag.batch_process_folder(folder_path, model_size)
            
            elif choice == '4':
                query = input("🔍 検索クエリを入力: ").strip()
                if query:
                    rag.search_and_display(query)
            
            elif choice == '5':
                rag.show_database_stats()
            
            elif choice == '6':
                rag.show_transcript_details()
            
            elif choice == '7':
                rag.open_category_folder()
            
            elif choice == '8':
                rag.show_processing_stats()
            
            elif choice == '9':
                rag.show_learned_patterns()
            
            elif choice == '10':
                rag.clear_learned_patterns()
            
            elif choice == '11':
                rag.claude_integration.manage_api_key()
            
            elif choice == '12':
                rag.show_help_and_examples()
            
            elif choice == '13':
                print("🎉 システムを終了します")
                rag.show_processing_stats()  # 最終統計表示
                break
            
            else:
                print("❌ 無効な選択です")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ ユーザーによる中断")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("📦 必要なライブラリ:")
    print("pip install whisper yt-dlp numpy scikit-learn sentence-transformers requests")
    print()
    
    main()
