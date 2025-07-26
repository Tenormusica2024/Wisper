import os
import warnings
import json
from datetime import datetime
from pathlib import Path
import tempfile

# 必要なライブラリのインポート
try:
    import whisper
    import yt_dlp
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"必要なライブラリが不足しています: {e}")
    print("\n以下のコマンドでインストールしてください:")
    print("pip install openai-whisper yt-dlp scikit-learn sentence-transformers torch")
    exit(1)

warnings.filterwarnings("ignore")

class VoiceRAGSystem:
    def __init__(self, db_path="voice_rag_db.json"):
        """
        音声RAGシステムの初期化
        """
        self.db_path = db_path
        self.whisper_model = None
        self.sentence_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.database = self.load_database()
        
        # カテゴリ分類用のキーワード辞書
        self.category_keywords = {
            'ビジネス': ['会議', '売上', 'マーケティング', '戦略', '予算', 'プロジェクト', '顧客', '営業'],
            '技術': ['プログラミング', 'AI', 'システム', 'データベース', 'アルゴリズム', 'API', 'クラウド'],
            '教育': ['学習', '授業', '講義', '試験', '研究', '論文', '学生', '教育'],
            'エンターテイメント': ['映画', '音楽', 'ゲーム', 'アニメ', 'スポーツ', '芸能', '娯楽'],
            '日常': ['家族', '友人', '買い物', '料理', '旅行', '健康', '趣味'],
            'ニュース': ['政治', '経済', '社会', '国際', '事件', '災害', '政府']
        }
    
    def load_whisper_model(self, model_size="small"):
        """
        Whisperモデルの読み込み
        small: より高精度 (244MB)
        base: 標準 (142MB) 
        tiny: 軽量だが精度低 (39MB)
        """
        print(f"Whisperモデル ({model_size}) を読み込み中...")
        self.whisper_model = whisper.load_model(model_size)
        print("Whisperモデルの読み込み完了")
    
    def load_sentence_model(self):
        """
        文埋め込みモデルの読み込み
        """
        print("文埋め込みモデルを読み込み中...")
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("文埋め込みモデルの読み込み完了")
    
    def download_youtube_audio(self, url, output_path=None):
        """
        YouTubeから音声を取得
        """
        if output_path is None:
            output_path = tempfile.mkdtemp()
        
        # ffmpegの存在確認
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("エラー: ffmpegが見つかりません。")
            print("解決方法:")
            print("1. 管理者としてPowerShellを実行: choco install ffmpeg")
            print("2. 手動インストール: https://ffmpeg.org/download.html")
            print("3. wingetを使用: winget install FFmpeg")
            print("4. 先にmp3ファイルでテストしてください")
            return None, None
        
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
                
                # ダウンロードされたファイルを探す
                for file in os.listdir(output_path):
                    if file.endswith('.mp3'):
                        return os.path.join(output_path, file), title
                        
        except Exception as e:
            print(f"YouTube音声取得エラー: {e}")
            print("ffmpegが正しくインストールされているか確認してください")
            return None, None
    
    def transcribe_audio(self, audio_path):
        """
        音声をテキストに変換
        """
        if self.whisper_model is None:
            self.load_whisper_model()
        
        print("音声をテキストに変換中...")
        try:
            # より高精度な設定
            result = self.whisper_model.transcribe(
                audio_path, 
                language='ja',
                task='transcribe',
                temperature=0.0,  # より安定した結果
                best_of=5,        # より良い結果を選択
                beam_size=5       # ビームサーチで精度向上
            )
            return result['text']
        except Exception as e:
            print(f"音声変換エラー: {e}")
            return None
    
    def classify_text(self, text):
        """
        テキストをカテゴリ分類
        """
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword.lower())
            category_scores[category] = score
        
        # 最高スコアのカテゴリを返す
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return "その他"
    
    def create_embedding(self, text):
        """
        テキストの埋め込みベクトルを作成
        """
        if self.sentence_model is None:
            self.load_sentence_model()
        
        return self.sentence_model.encode([text])[0].tolist()
    
    def add_to_database(self, source_type, source_path, title, transcript, category):
        """
        データベースにエントリを追加
        """
        embedding = self.create_embedding(transcript)
        
        entry = {
            'id': len(self.database) + 1,
            'source_type': source_type,  # 'youtube' or 'file'
            'source_path': source_path,
            'title': title,
            'transcript': transcript,
            'category': category,
            'embedding': embedding,
            'created_at': datetime.now().isoformat(),
            'word_count': len(transcript.split()),
            'char_count': len(transcript)
        }
        
        self.database.append(entry)
        self.save_database()
        
        print(f"データベースに追加: {title} (カテゴリ: {category})")
    
    def load_database(self):
        """
        データベースを読み込み
        """
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_database(self):
        """
        データベースを保存
        """
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, ensure_ascii=False, indent=2)
    
    def search_similar(self, query, top_k=3):
        """
        類似検索
        """
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
        
        # 類似度でソートして上位k件を返す
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def process_youtube_url(self, url):
        """
        YouTube URLを処理
        """
        print(f"YouTube URL を処理中: {url}")
        
        # 音声ダウンロード
        audio_path, title = self.download_youtube_audio(url)
        if not audio_path:
            print("YouTube音声の取得に失敗しました")
            return False
        
        try:
            # 音声をテキスト化
            transcript = self.transcribe_audio(audio_path)
            if not transcript:
                print("音声変換に失敗しました")
                return False
            
            # カテゴリ分類
            category = self.classify_text(transcript)
            
            # データベースに追加
            self.add_to_database('youtube', url, title, transcript, category)
            
            print(f"処理完了: {title}")
            print(f"カテゴリ: {category}")
            print(f"テキスト長: {len(transcript)}文字")
            
            return True
            
        finally:
            # 一時ファイルを削除
            try:
                os.remove(audio_path)
            except:
                pass
    
    def process_audio_file(self, file_path):
        """
        音声ファイルを処理
        """
        if not os.path.exists(file_path):
            print(f"ファイルが存在しません: {file_path}")
            return False
        
        print(f"音声ファイルを処理中: {file_path}")
        
        # 音声をテキスト化
        transcript = self.transcribe_audio(file_path)
        if not transcript:
            print("音声変換に失敗しました")
            return False
        
        # カテゴリ分類
        category = self.classify_text(transcript)
        
        # ファイル名をタイトルとして使用
        title = os.path.basename(file_path)
        
        # データベースに追加
        self.add_to_database('file', file_path, title, transcript, category)
        
        print(f"処理完了: {title}")
        print(f"カテゴリ: {category}")
        print(f"テキスト長: {len(transcript)}文字")
        
        return True
    
    def search_and_display(self, query):
        """
        検索結果を表示
        """
        print(f"\n検索クエリ: '{query}'")
        print("-" * 50)
        
        results = self.search_similar(query)
        
        if not results:
            print("検索結果がありません")
            return
        
        for i, (entry, similarity) in enumerate(results, 1):
            print(f"\n{i}. {entry['title']}")
            print(f"   カテゴリ: {entry['category']}")
            print(f"   類似度: {similarity:.3f}")
            print(f"   ソース: {entry['source_type']}")
            print(f"   作成日: {entry['created_at']}")
            print(f"   内容抜粋: {entry['transcript'][:100]}...")
    
    def show_database_stats(self):
        """
        データベースの統計情報を表示
        """
        if not self.database:
            print("データベースは空です")
            return
        
        print(f"\n=== データベース統計 ===")
        print(f"総エントリ数: {len(self.database)}")
        
        # カテゴリ別統計
        categories = {}
        total_words = 0
        
        for entry in self.database:
            category = entry['category']
            categories[category] = categories.get(category, 0) + 1
            total_words += entry['word_count']
        
        print(f"総単語数: {total_words:,}")
        print(f"平均単語数: {total_words//len(self.database):,}")
        
        print("\nカテゴリ別分布:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}件")
    
    def show_transcript_details(self):
        """
        文字起こし内容の詳細表示
        """
        if not self.database:
            print("データベースは空です")
            return
        
        print(f"\n=== 保存されている文字起こし一覧 ===")
        for i, entry in enumerate(self.database, 1):
            print(f"\n{i}. {entry['title']}")
            print(f"   カテゴリ: {entry['category']}")
            print(f"   ソース: {entry['source_type']}")
            print(f"   作成日: {entry['created_at']}")
            print(f"   文字数: {len(entry['transcript'])}文字")
        
        try:
            choice = int(input(f"\n詳細を見たい項目番号 (1-{len(self.database)}): "))
            if 1 <= choice <= len(self.database):
                entry = self.database[choice - 1]
                print(f"\n" + "="*60)
                print(f"タイトル: {entry['title']}")
                print(f"カテゴリ: {entry['category']}")
                print(f"ソース: {entry['source_path']}")
                print(f"作成日時: {entry['created_at']}")
                print(f"文字数: {len(entry['transcript'])}文字")
                print(f"単語数: {entry['word_count']}語")
                print(f"\n【文字起こし内容】")
                print("-" * 60)
                print(entry['transcript'])
                print("-" * 60)
            else:
                print("無効な番号です")
        except ValueError:
            print("数字を入力してください")


def main():
    """
    メイン実行関数
    """
    rag = VoiceRAGSystem()
    
    while True:
        print("\n" + "="*60)
        print("音声テキスト化＆分類RAGシステム")
        print("="*60)
        print("1. YouTube URLから音声を処理")
        print("2. 音声ファイルを処理")
        print("3. テキスト検索")
        print("4. データベース統計表示")
        print("5. 文字起こし内容の詳細表示")
        print("6. 終了")
        
        choice = input("\n選択してください (1-6): ").strip()
        
        if choice == '1':
            url = input("YouTube URL を入力してください: ").strip()
            if url:
                rag.process_youtube_url(url)
        
        elif choice == '2':
            file_path = input("音声ファイルのパスを入力してください: ").strip()
            if file_path:
                rag.process_audio_file(file_path)
        
        elif choice == '3':
            query = input("検索クエリを入力してください: ").strip()
            if query:
                rag.search_and_display(query)
        
        elif choice == '4':
            rag.show_database_stats()
        
        elif choice == '5':
            rag.show_transcript_details()
        
        elif choice == '6':
            print("システムを終了します")
            break
        
        else:
            print("無効な選択です")


if __name__ == "__main__":
    main()
