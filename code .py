#المكتبات 
import pandas as pd
from collections import Counter
from collections import defaultdict
from collections import deque
import re
from typing import Optional

#قراءة الملف مع اظهار جميع الصفوف والاعمدة دون اختصار 
df = pd.read_csv(r"C:\Users\ASUS\Downloads\archive (11)data\train.csv", encoding='latin1')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# تنظيف البيانات 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"[^a-z\s]", "", text)  
    text = re.sub(r"\s+", " ", text)  
    return text.strip()

# معالجة البيانات 
def preprocess(text):
    if pd.isna(text):
        return []
    text = clean_text(text)
    return text.split()


df['cleaned_words'] = df['text'].apply(preprocess)
df['cleaned_text'] = df['cleaned_words'].apply(lambda words: ' '.join(words))

# تحليل الكلمات 
def word_statistics(df):
    all_words = [word for words in df['cleaned_words'] for word in words]  
    unique_words = set(all_words)  
    word_counts = Counter(all_words)  

    print(f"\nTotal Words: {len(all_words)}")
    print(f"Unique Words: {len(unique_words)}")
    print("\nTop 10 Most Frequent Words:")
    for word, freq in word_counts.most_common(10):
        print(f" - {word}: {freq}")
    return {
        'all_words': all_words,
        'unique_words': unique_words,
        'word_counts': word_counts }

#linked list node 
class CharNode:
    def __init__(self, value):
        self.value = value
        self.next: Optional['CharNode'] = None

#  linked list تحويل النص إلى 
def build_linked_list(text):
    head = CharNode(text[0]) if text else None
    current = head
    for ch in text[1:]:
        new_node = CharNode(ch)
        current.next = new_node
        current = new_node
    return head

#linked list و التمثيل في queue ومعالجنها في  stack تحليل الحروف وتخزينها في
def character_statistics(df):
    all_text = "".join(df['cleaned_text'].fillna(''))
    unique_chars = set(all_text)
    char_freq = Counter(all_text)
   
    char_stack = list(all_text)  
    char_queue = deque(all_text)  
    linked_list_head = build_linked_list(all_text)

    node = linked_list_head
    for _ in range(10):
        if not node: break
        print(f"'{node.value}' -> ", end="")
        node = node.next
  
    print("\n Character Statistics Summary")
    print(f" Total Characters: {len(all_text)}")
    print(f"Unique Characters: {len(unique_chars)} => {sorted(unique_chars)}")
    print("\nTop 10 Most Frequent Characters:")
    for char, count in char_freq.most_common(20):
        print(f"   - '{char}': {count}")

    print("\n Stack Example (last char):", char_stack[-1] if char_stack else "Empty")
    print(" Queue Example (first char):", char_queue[0] if char_queue else "Empty")
    
    print("\n Linked List First 10 Chars:")
    node = linked_list_head
    for _ in range(10):
        if not node: break
        print(f"'{node.value}' -> ", end="")
        node = node.next
    print("None")

    return {
        'all_text': all_text,
        'unique_chars': unique_chars,
        'char_counts': char_freq,
        'char_stack': char_stack,
        'char_queue': char_queue,
        'linked_list_head': linked_list_head
    }

#  stack و تخزن في queue البحث عن كلمة في النصوص تعالج في
search_history_stack = []
search_queue = deque()
def search_word_summary(df, targets):

    if isinstance(targets, str):
        targets = [targets] # اذا كانت كلمة واحدة تحول الى قائمة 

    for t in targets:
        search_queue.append(t.lower())

    all_results = []
    while search_queue:
        target = search_queue.popleft()
        count = 0
        locations = []
        rows_with_target = set()

        for idx, words in df['cleaned_words'].items():
            if target in words:
                word_count = words.count(target)
                count += word_count
                locations.append(idx)
                rows_with_target.add(idx)
        
        print(f"\n Word Search: '{target}'")
        print(f" Total Occurrences: {count}")
        print(f"Found in {len(locations)} row(s).")
        if locations:
            print(f"Row Indexes: {locations}")
        else:
            print("No occurrences found.")
    
        #  stack تخزين النتيجة في 
        result = {
            'word': target,
            'total_count': count,
            'locations': locations,
            'unique_rows': rows_with_target
        }
        search_history_stack.append(result)
        all_results.append(result)

    return all_results  

#استبدال كلمة بكلمة جديدة 
replace_history_stack = []
def replace_word(df, old_word, new_word):
    old_word, new_word = old_word.lower(), new_word.lower()
    replace_history_stack.append(df['cleaned_words'].copy())
    df['cleaned_words'] = df['cleaned_words'].apply( lambda words: [new_word if word == old_word else word for word in words] )
    df['cleaned_text'] = df['cleaned_words'].apply(lambda words: ' '.join(words))
    print(f"\n Replaced '{old_word}' with '{new_word}' successfully.")
    return df  

# بناء نموذج Bigram و Trigram
def build_ngram_models(df):
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)

    for words in df['cleaned_words']:
        for i in range(len(words) - 1):
            bigrams[words[i]][words[i + 1]] += 1

        for i in range(len(words) - 2):
            trigrams[(words[i], words[i + 1])][words[i + 2]] += 1
    return bigrams, trigrams


# التنبؤ بالكلمة التالية 
prediction_history_stack = []
def predict_next_word(bigrams, trigrams, input_text):
    words = input_text.lower().strip().split()
    candidates = {}
    if len(words) == 1:
        candidates = bigrams.get(words[0], {})
    elif len(words) >= 2:
        candidates = trigrams.get((words[-2], words[-1]), {})

    if candidates:
        top_predictions = candidates.most_common(5)
        print(f"\nSuggestions for '{input_text}':")
        for word, count in top_predictions:
            print(f" - {word} ({count})")
        prediction_history_stack.append({'input': input_text,'predictions': top_predictions })
        return top_predictions
    else:
        print("No suggestions found.")
        return []

# تنظيف وتحليل نص واحد
positive_words = {"love", "happy", "great", "good", "awesome", "excellent", "fantastic", "amazing",
"fun", "smile", "hope", "peace", "like", "joy", "nice", "cool", "wonderful", "cute"}

negative_words = {"sad", "bad", "hate", "angry", "terrible", "awful", "worse", "worst", "sucks",
"depressing", "cry", "pain", "problem", "annoying", "mad", "bully", "hurt", "poor"}

def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())
def analyze_text(text):
    words = preprocess(text)
    pos = sum(w in positive_words for w in words)
    neg = sum(w in negative_words for w in words)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    elif pos == neg and pos > 0:
        return "Mixed"
    else:
        return "Neutral"

# تحليل جميع النصوص باستخدام Queue
def analyze_with_queue(df):
    tweet_queue = deque(df['text'].fillna('').tolist())
    sentiments = []
    while tweet_queue:
        text = tweet_queue.popleft()
        sentiments.append(analyze_text(text))

    df['sentiment'] = sentiments
    print(df[['text', 'sentiment']].head())
    return df

# تحليل صف معين بواسطة المستخدم
def analyze_row_by_index(df):

    try:
        index = int(input("Enter row number to analyze: "))
        if index < 0 or index >= len(df):
            print("Error: invalid index, please try again.")
            return
        text = df.loc[index, 'text']
        sentiment = analyze_text(text)
        print(f"Text at index {index}:\n{text}")
        print(f"\nPredicted Sentiment: {sentiment}")
    except ValueError:
        print("Error: please enter a valid integer.")


def main():
    print("=" * 60)
    print("Welcome to The Intelligent Text Processor")
    print("=" * 60)

    # تحميل النموذج اللغوي
    bigrams, trigrams = build_ngram_models(df)

    while True:
        print("\n Main Menu:")
        print("1. Word Statistics")
        print("2. Character Statistics")
        print("3. Search for a Word")
        print("4. Replace a Word")
        print("5. Analyze All Sentiments")
        print("6. Predict Next Word")
        print("7. Analyze Single Row Sentiment")
        print("8. Exit")
        choice = input("Enter your choice (1-8): ")
       
        if choice == '1':
            word_statistics(df)
        
        elif choice == '2':
            character_statistics(df)
        
        elif choice == '3':
            word = input("Enter word to search: ")
            search_word_summary(df, word)
       
        elif choice == '4':
            old = input("Word to replace: ")
            new = input("Replace with: ")
            replace_word(df, old, new)
       
        elif choice == '5':
            analyze_with_queue(df)
        
        elif choice == '6':
            phrase = input("Enter phrase: ")
            predict_next_word(bigrams, trigrams, phrase)
       
        elif choice == '7':
            analyze_row_by_index(df)
        
        elif choice == '8':
            print("=" * 60)
            print("\n Thank you for using The Intelligent Text Processor good bye")
            print("=" * 60)
            break

        else:
            print("Invalid choice")

# تشغيل 
if __name__ == "__main__":
    main()          
