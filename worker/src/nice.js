// NICE CKS hypertension topic — page list and HTML extraction.
// The pages are server-rendered Gatsby; content lives in <main>, organised
// under h2/h3 headings. No DOM parser is available in Workers, so this is a
// deliberately conservative tag-stripping extractor.

// Content pages only — meta pages (references, "how this topic was
// developed", …) are excluded from the corpus.
export const CONTENT_PATHS = [
  "/topics/hypertension/",
  "/topics/hypertension/background-information/definition/",
  "/topics/hypertension/background-information/prevalence/",
  "/topics/hypertension/background-information/risk-factors/",
  "/topics/hypertension/background-information/secondary-causes-of-hypertension/",
  "/topics/hypertension/background-information/complications-prognosis/",
  "/topics/hypertension/diagnosis/diagnosis/",
  "/topics/hypertension/diagnosis/investigations/",
  "/topics/hypertension/management/management/",
  "/topics/hypertension/goals-outcome-measures/",
  "/topics/hypertension/prescribing-information/angiotensin-converting-enzyme-inhibitors/",
  "/topics/hypertension/prescribing-information/angiotensin-ii-receptor-blockers/",
  "/topics/hypertension/prescribing-information/calcium-channel-blockers/",
  "/topics/hypertension/prescribing-information/thiazide-like-diuretics/",
  "/topics/hypertension/prescribing-information/spironolactone/",
  "/topics/hypertension/prescribing-information/alpha-blockers/",
  "/topics/hypertension/prescribing-information/beta-blockers/",
  "/topics/hypertension/prescribing-information/covid-19/",
];

const ENTITIES = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#x27;": "'",
  "&#39;": "'",
  "&nbsp;": " ",
  "&ndash;": "–",
  "&mdash;": "—",
};

function decodeEntities(s) {
  return s
    .replace(/&(amp|lt|gt|quot|nbsp|ndash|mdash);|&#x27;|&#39;/g, (m) => ENTITIES[m] ?? m)
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, n) => String.fromCharCode(parseInt(n, 16)));
}

function stripTags(html) {
  return decodeEntities(
    html
      .replace(/<(script|style|nav|svg)[\s\S]*?<\/\1>/gi, " ")
      // Block-level closes become newlines so list items keep separation.
      .replace(/<\/(p|li|ul|ol|div|tr|table|h[1-6])>/gi, "\n")
      .replace(/<li[^>]*>/gi, "• ")
      .replace(/<[^>]+>/g, " "),
  )
    .replace(/[ \t]+/g, " ")
    .replace(/ ?\n ?/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// Split a page's <main> content into { heading, text } sections on h2/h3.
export function extractSections(html) {
  const titleMatch = html.match(/<title[^>]*>([^<]*)<\/title>/i);
  const title = decodeEntities(titleMatch?.[1] || "")
    .replace(/\s*\|\s*NICE.*$/i, "")
    .replace(/\s*-\s*CKS\s*$/i, "")
    .trim();

  const mainMatch = html.match(/<main[^>]*>([\s\S]*?)<\/main>/i);
  const main = mainMatch ? mainMatch[1] : html;

  const parts = main.split(/(?=<h[23][^>]*>)/i);
  const sections = [];
  for (const part of parts) {
    const headingMatch = part.match(/^<h[23][^>]*>([\s\S]*?)<\/h[23]>/i);
    const heading = headingMatch ? stripTags(headingMatch[1]).replace(/\n/g, " ").trim() : "";
    const bodyHtml = headingMatch ? part.slice(headingMatch[0].length) : part;
    const text = stripTags(bodyHtml);
    // Skip navigation stubs and boilerplate-only fragments.
    if (text.length < 120) continue;
    sections.push({ heading, text });
  }
  return { title, sections };
}

// Group section text into chunks of roughly CHUNK_CHARS, splitting long
// sections on paragraph boundaries.
const CHUNK_CHARS = 1800;

export function chunkSections(sections) {
  const chunks = [];
  for (const { heading, text } of sections) {
    if (text.length <= CHUNK_CHARS) {
      chunks.push({ section: heading, text });
      continue;
    }
    const paragraphs = text.split(/\n\n+/);
    let current = "";
    for (const p of paragraphs) {
      if (current && current.length + p.length + 2 > CHUNK_CHARS) {
        chunks.push({ section: heading, text: current.trim() });
        current = "";
      }
      // A single paragraph longer than the budget is split hard.
      if (p.length > CHUNK_CHARS) {
        for (let i = 0; i < p.length; i += CHUNK_CHARS) {
          chunks.push({ section: heading, text: p.slice(i, i + CHUNK_CHARS).trim() });
        }
      } else {
        current = current ? `${current}\n\n${p}` : p;
      }
    }
    if (current.trim()) chunks.push({ section: heading, text: current.trim() });
  }
  return chunks.filter((c) => c.text.length >= 120);
}
