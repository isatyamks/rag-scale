
import logging
from pathlib import Path
from typing import Optional
import wikipediaapi

class WikiLoader:

    def __init__(self, lang: str = "en"):
        self.wiki = wikipediaapi.Wikipedia(
            language=lang,
            user_agent="RAG-Scale/1.0 (https://github.com/isatyamks/RAG)",
            extract_format=wikipediaapi.ExtractFormat.HTML
        )

    def _extract_sections(self, sections, level=0):
        content = []
        for section in sections:
            if section.title:
                content.append(f"\n{'#' * (level + 1)} {section.title}\n")
            if section.text:
                text_lines = [line.strip() for line in section.text.split('\n') if line.strip()]
                if text_lines:
                    content.append('\n'.join(text_lines) + '\n')
            if section.sections:
                content.extend(self._extract_sections(section.sections, level + 1))
        return content

    def fetch_page(self, topic: str, output_dir: Path) -> Optional[Path]:
        page = self.wiki.page(topic)
        
        if not page.exists():
            logging.warning(f"Wikipedia page '{topic}' does not exist.")
            return None

        logging.info(f"Fetching Wikipedia page: {page.title}")
        
        content_parts = []
        
        content_parts.append(f"# {page.title}\n\n")
        
        if page.summary:
            content_parts.append(f"{page.summary}\n\n")
        
        if page.sections:
            section_content = self._extract_sections(page.sections)
            content_parts.extend(section_content)
        
        full_content = "".join(content_parts)
        
        if not full_content.strip() or len(full_content) < 100:
            logging.warning(f"Wikipedia page '{topic}' has insufficient text.")
            return None

        safe_filename = "".join(x for x in topic if (x.isalnum() or x in "._- ")).strip()
        file_path = output_dir / f"{safe_filename}.txt"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)
            
        logging.info(f"Saved Wikipedia content to {file_path} ({len(full_content)} chars)")
        return file_path

