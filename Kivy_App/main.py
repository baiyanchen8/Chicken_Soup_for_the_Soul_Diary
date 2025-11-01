import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.slider import Slider

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === 初始化模型與資料 ===
df = pd.read_csv("chicken_soup_with_features.csv")
vectors = torch.load("chicken_soup_vectors.pt")
df['vector'] = vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# === 搜尋函式 ===
def get_recommendation(mood_text, top_k=5):
    mood_vector = model.encode(mood_text, convert_to_tensor=True)
    df['similarity'] = df['vector'].apply(lambda x: util.cos_sim(x, mood_vector).item())
    results = df.sort_values(by='similarity', ascending=False).head(top_k)
    return results[['text', 'similarity']]

# === Kivy 介面 ===
class MoodApp(App):
    def build(self):
        Window.size = (400, 700)
        self.title = "雞湯精靈 🍵"
        layout = BoxLayout(orientation='vertical', padding=15, spacing=10)

        self.labels = {}
        self.sliders = {}
        categories = {
            "壓力": "stress",
            "開心": "happiness",
            "幽默": "humor",
            "鼓勵需求": "encouragement"
        }

        for label, key in categories.items():
            row = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
            lbl = Label(text=f"{label} (1~5)", size_hint_x=0.4)
            slider = Slider(min=1, max=5, value=3, step=1)
            self.labels[key] = lbl
            self.sliders[key] = slider
            row.add_widget(lbl)
            row.add_widget(slider)
            layout.add_widget(row)

        self.result_box = BoxLayout(orientation='vertical', size_hint_y=None)
        self.result_scroll = ScrollView(size_hint=(1, 1))
        self.result_scroll.add_widget(self.result_box)

        self.btn = Button(text="🧠 生成雞湯建議", size_hint_y=None, height=50, on_press=self.generate_chickensoup)
        layout.add_widget(self.btn)
        layout.add_widget(self.result_scroll)
        return layout

    def generate_chickensoup(self, instance):
        stress = int(self.sliders['stress'].value)
        happy = int(self.sliders['happiness'].value)
        humor = int(self.sliders['humor'].value)
        encourage = int(self.sliders['encouragement'].value)

        mood_text = f"壓力:{stress}, 開心:{happy}, 幽默:{humor}, 鼓勵需求:{encourage}"
        results = get_recommendation(mood_text)

        self.result_box.clear_widgets()
        for _, row in results.iterrows():
            msg = f"💬 {row['text']}\n(相似度: {row['similarity']:.3f})"
            self.result_box.add_widget(Label(text=msg, size_hint_y=None, height=100))

# === 啟動應用 ===
if __name__ == "__main__":
    MoodApp().run()
