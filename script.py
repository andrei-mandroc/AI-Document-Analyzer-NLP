import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForQuestionAnswering

class NERExtractor:

    def __init__(self, model_name="dslim/bert-base-NER"):
        print(f"Se incarca modelul pentru extragerea entitatilor ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def extract_entities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

        entities = {"Persoane": [], "Locatii": [], "Organizatii": []}
        current_word = ""
        current_type = None

        for token, pred_id in zip(tokens, predictions):
            label = self.id2label[pred_id]
            if token in ["[CLS]", "[SEP]", "[PAD]"]: continue

            is_subword = token.startswith("##")
            clean_token = token.replace("##", "")

            if label != "O" or is_subword:
                if is_subword or label.startswith("I-"):
                    connector = "" if is_subword else " "
                    current_word += connector + clean_token
                elif label.startswith("B-"):
                    if current_word and current_type:
                        self._add_to_dict(entities, current_word, current_type)
                    current_word = clean_token
                    current_type = label[2:]
            else:
                if current_word and current_type:
                    self._add_to_dict(entities, current_word, current_type)
                current_word = ""
                current_type = None

        # eliminam entitatile prea scurte(mai putin de 2 caractere) si duplicatele
        for key in entities:
            entities[key] = list(set([e.strip() for e in entities[key] if len(e.strip()) > 2]))

        return entities

    def _add_to_dict(self, entities_dict, word, ent_type):
        if ent_type == "PER":
            entities_dict["Persoane"].append(word)
        elif ent_type == "LOC":
            entities_dict["Locatii"].append(word)
        elif ent_type == "ORG":
            entities_dict["Organizatii"].append(word)


class FileManager:

    #permite preluarea oricarui fisier text si salvarea rapoartelor text.

    def __init__(self, input_filepath, output_filepath="raport_final.txt"):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def read_text(self):
        if not os.path.exists(self.input_filepath):
            raise FileNotFoundError(f"Eroare: Fisierul {self.input_filepath} nu a fost gasit!")

        with open(self.input_filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def save_report(self, content):
        with open(self.output_filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"\n[SUCCES] Raportul a fost salvat in: {self.output_filepath}")


class ZeroShotClassifier:

    def __init__(self, model_name="facebook/bart-large-mnli"):
        print(f"[INFO] Se incarca modulul de clasificare domenii ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def classify(self, text,
                 candidate_labels=["Politics", "Economy", "Technology", "Environment", "Health", "Justice"]):

        text_snippet = text[:500]

        results = {}

        for label in candidate_labels:

            hypothesis = f"This text is about {label}."

            inputs = self.tokenizer(text_snippet, hypothesis, return_tensors="pt", truncation=True)

            with torch.no_grad():
                outputs = self.model(**inputs)


            logits = outputs.logits
            entailment_logit = logits[:, 2]
            results[label] = entailment_logit.item()

        raw_scores = torch.tensor(list(results.values()))
        probs = torch.nn.functional.softmax(raw_scores, dim=0)

        final_results = {label: prob.item() * 100 for label, prob in zip(results.keys(), probs)}

        sorted_labels = sorted(final_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_labels[0]

class TextStatistics:

    def __init__(self, text):
        self.raw_text = text
        self.clean_text = self._clean_data()

    def _clean_data(self):
        text = re.sub(r'\s+', ' ', self.raw_text)
        return text.strip()

    def count_characters(self):
        return len([char for char in self.clean_text if char.isalnum()])

    def count_words(self):
        words = self.clean_text.split()
        return len(words)

    def get_all_stats(self):
        return {
            "Numar caractere": self.count_characters(),
            "Numar total cuvinte": self.count_words(),
            "Lungime text brut (cu spatii)": len(self.raw_text)
        }

    def get_keywords(self, top_n=5):
        stop_words = {"the", "and", "is", "of", "to", "in", "that", "was", "for", "on", "with", "as", "by", "at", "it",
                      "from"}

        words = re.findall(r'\w+', self.clean_text.lower())

        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]

        counts = {}
        for w in filtered_words:
            counts[w] = counts.get(w, 0) + 1

        sorted_keywords = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_keywords[:top_n]]



class LanguageDetector:

    def __init__(self, model_name="qanastek/51-languages-classifier"):
        print(f"[INFO] Se incarca modelul pentru detectarea limbii ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def detect_language(self, text):

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

        language_label = self.model.config.id2label[predicted_class_id]
        return language_label.upper()


class StyleAnalyzer:

    def __init__(self, model_name="facebook/bart-large-mnli"):
        print(f"[INFO] Se incarca analizorul de stil ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_style(self, text):

        style_labels = ["Formal", "Informal", "Urgent", "Analytical", "Narrative"]

        snippet = text[:512]

        style_scores = {}
        for label in style_labels:
            hypothesis = f"This text is written in a {label} style."
            inputs = self.tokenizer(snippet, hypothesis, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)

            score = outputs.logits[0, 2].item()
            style_scores[label] = score

        probs = torch.nn.functional.softmax(torch.tensor(list(style_scores.values())), dim=0)
        final_styles = {label: prob.item() * 100 for label, prob in zip(style_scores.keys(), probs)}

        best_style = max(final_styles, key=final_styles.get)
        return best_style, final_styles[best_style]


class ManualSummarizer:

    def __init__(self, model_name="facebook/bart-large-cnn"):
        print(f"[INFO] Se incarca modelul pentru sumarizare ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text, max_length=150):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                min_length=40,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


class QuestionAnswering:

    def __init__(self, model_name="deepset/roberta-base-squad2"):
        print(f"[INFO] Se incarca asistentul interactiv (QA) ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def answer_question(self, question, context):

        inputs = self.tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        if answer_start_index >= answer_end_index:
            return "Nu am gasit un raspuns clar in text."

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

        return answer

class NLPProjectApp:

    def __init__(self, input_file):
        self.file_manager = FileManager(input_filepath=input_file)
        self.text = self.file_manager.read_text()

        self.stats_engine = TextStatistics(self.text)
        self.lang_detector = LanguageDetector()
        self.ner_extractor = NERExtractor()
        self.summarizer = ManualSummarizer()
        self.qa_bot = QuestionAnswering()
        self.domain_classifier = ZeroShotClassifier()
        self.style_analyzer = StyleAnalyzer()

    def run_analysis(self):

        print("-> Se calculeaza statisticile...")
        stats = self.stats_engine.get_all_stats()

        print("-> Se detecteaza limba...")
        limba = self.lang_detector.detect_language(self.stats_engine.clean_text)

        print("-> Se scaneaza documentul pentru entitati...")
        entitati = self.ner_extractor.extract_entities(self.stats_engine.clean_text)

        print("-> Se genereaza sumarul...")
        if stats["Numar total cuvinte"] > 30:
            sumar = self.summarizer.summarize(self.stats_engine.clean_text)
        else:
            sumar = "Textul este prea scurt pentru a fi sumarizat eficient."

        domeniu, scor = self.domain_classifier.classify(self.stats_engine.clean_text)
        stil, scor_stil = self.style_analyzer.analyze_style(self.text)
        cuvinte_cheie = self.stats_engine.get_keywords(5)


        raport = "\n\n"
        raport += "       RAPORT SCANARE DOCUMENTE      \n\n"


        raport += f"1. ANALIZA TEMATICA:\n"
        raport += f"   - Domeniul principal:  {domeniu.upper()}  ({scor:.1f}%)\n"
        raport += f"   - Stil:  {stil.upper()} ({scor_stil:.1f}%)\n"
        raport += f"   - Limba detectata: {limba}\n\n"

        raport += "2. STATISTICI TEXT:\n"
        raport += f"   - Cuvinte cheie: {', '.join(cuvinte_cheie)}\n"
        for key, value in stats.items():
            raport += f"   - {key}: {value}\n"

        raport += "\n3. DATE DE INTERES EXTRASE:\n"
        raport += f"   -  Persoane: {', '.join(entitati['Persoane']) if entitati['Persoane'] else 'Niciuna detectata'}\n"
        raport += f"   -  Locatii: {', '.join(entitati['Locatii']) if entitati['Locatii'] else 'Niciuna detectata'}\n"
        raport += f"   -  Organizatii: {', '.join(entitati['Organizatii']) if entitati['Organizatii'] else 'Niciuna detectata'}\n"

        raport += f"\n4. TEXT ORIGINAL (Fragment):\n   \"{self.text[:200]}...\"\n\n"
        raport += f"5. Rezumatul documentului:\n   {sumar}\n"


        print("\n" + raport)
        self.file_manager.save_report(raport)

        print("\n\n\n")
        print(" ASISTENTUL VIRTUAL ESTE ACTIV!")
        print("   Poti pune intrebari despre textul scanat.")

        while True:
            intrebare = input("\nIntrebarea: ")

            if intrebare.lower() in ['exit', 'stop', 'quit', 'gata']:
                print("O zi frumoasă!")
                break

            if intrebare.strip() == "":
                continue

            print("Se cauta raspunsul in document...")
            raspuns = self.qa_bot.answer_question(intrebare, self.stats_engine.clean_text)

            print(f"RASPUNS: {raspuns}")

if __name__ == "__main__":

    fisier_intrare = r"C:\Users\Andrei Mandroc\Desktop\articol.txt"

    app = NLPProjectApp(input_file=fisier_intrare)
    app.run_analysis()