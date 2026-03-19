//! CMU Pronouncing Dictionary loader — converts ARPABET to IPA and populates
//! the pronunciation trie with 135K English word pronunciations.

use crate::pronunciation::PronunciationTrie;
use std::path::Path;

/// Load a CMUdict file and populate a PronunciationTrie.
///
/// CMUdict format: `word  PH1 PH2 PH3` (ARPABET phonemes, stress digits on vowels).
/// Alternate pronunciations appear as `word(2)`, `word(3)`, etc.
pub fn load_cmudict(path: &Path) -> Result<PronunciationTrie, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read cmudict: {}", e))?;

    let mut trie = PronunciationTrie::new();
    let mut loaded = 0u32;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }

        // Split word from phonemes. Format: "word  PH1 PH2 PH3"
        // Some lines use single space, some double.
        let (word_raw, phonemes_str) = match line.find(|c: char| c.is_ascii_whitespace()) {
            Some(idx) => (&line[..idx], line[idx..].trim()),
            None => continue,
        };

        // Strip variant markers like "(2)", "(3)".
        let word = word_raw
            .split('(')
            .next()
            .unwrap_or(word_raw)
            .to_lowercase();

        // Skip entries with apostrophes at the start (contractions like 'bout).
        if word.starts_with('\'') {
            continue;
        }

        // Convert ARPABET phonemes to IPA.
        let ipa = arpabet_to_ipa(phonemes_str);
        if ipa.is_empty() {
            continue;
        }

        let ipa_str = ipa.join("");
        // Use a default sense count of 1; common words get higher values below.
        trie.insert(&word, &ipa_str, word_frequency_estimate(&word));
        loaded += 1;
    }

    log::info!("Loaded {} pronunciations from CMUdict", loaded);
    Ok(trie)
}

/// Convert an ARPABET phoneme string to a sequence of IPA symbols.
///
/// Input: "AH0 B AE1 N D AH0 N" (ARPABET with stress digits)
/// Output: ["ə", "b", "æ", "n", "d", "ə", "n"]
fn arpabet_to_ipa(arpabet: &str) -> Vec<&'static str> {
    arpabet
        .split_whitespace()
        .filter_map(|phone| {
            // Strip stress digit (0, 1, 2) from vowels.
            let base = phone.trim_end_matches(|c: char| c.is_ascii_digit());
            let stress: u8 = phone
                .chars()
                .last()
                .and_then(|c| c.to_digit(10))
                .unwrap_or(0) as u8;

            arpabet_phone_to_ipa(base, stress)
        })
        .collect()
}

/// Map a single ARPABET phone to IPA.
fn arpabet_phone_to_ipa(phone: &str, stress: u8) -> Option<&'static str> {
    Some(match phone {
        // Vowels — stress 0 often reduces to schwa.
        "AA" => "ɑ",
        "AE" => "æ",
        "AH" => {
            if stress == 0 {
                "ə"
            } else {
                "ʌ"
            }
        }
        "AO" => "ɔ",
        "AW" => "aʊ",
        "AY" => "aɪ",
        "EH" => "ɛ",
        "ER" => {
            if stress == 0 {
                "ɜ"
            } else {
                "ɜː"
            }
        }
        "EY" => "eɪ",
        "IH" => "ɪ",
        "IY" => "iː",
        "OW" => "oʊ",
        "OY" => "ɔɪ",
        "UH" => "ʊ",
        "UW" => "uː",

        // Consonants.
        "B" => "b",
        "CH" => "tʃ",
        "D" => "d",
        "DH" => "ð",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "JH" => "dʒ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "P" => "p",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",

        _ => return None,
    })
}

/// Word frequency estimate based on corpus frequency data.
///
/// Returns a sense_count proxy ranging from 1 (rare) to 500 (most common).
/// The top ~2500 English words are explicitly ranked. Others use a length heuristic.
fn word_frequency_estimate(word: &str) -> u32 {
    // Use a static lookup for the most common English words (based on corpus frequency).
    // This is critical for disambiguation — without it, "sajak" competes with "the".
    if let Some(freq) = WORD_FREQUENCIES.iter().find_map(|&(w, f)| {
        if w == word { Some(f) } else { None }
    }) {
        return freq;
    }

    // Fallback: shorter words tend to be more common.
    match word.len() {
        1..=3 => 5,
        4..=6 => 2,
        7..=9 => 1,
        _ => 1,
    }
}

/// Top ~2500 English words by frequency, with approximate frequency ranks.
/// Frequencies are scaled: 500 = most common function words, 1 = rare.
static WORD_FREQUENCIES: &[(&str, u32)] = &[
    // Rank 1-50: ultra-high frequency function words (500).
    ("the", 500), ("be", 500), ("to", 500), ("of", 500), ("and", 500),
    ("a", 500), ("in", 500), ("that", 500), ("have", 500), ("i", 500),
    ("it", 500), ("for", 500), ("not", 500), ("on", 500), ("with", 500),
    ("he", 500), ("as", 500), ("you", 500), ("do", 500), ("at", 500),
    ("this", 500), ("but", 500), ("his", 500), ("by", 500), ("from", 500),
    ("they", 500), ("we", 500), ("say", 500), ("her", 500), ("she", 500),
    ("or", 500), ("an", 500), ("will", 500), ("my", 500), ("one", 500),
    ("all", 500), ("would", 500), ("there", 500), ("their", 500), ("what", 500),
    ("so", 500), ("up", 500), ("out", 500), ("if", 500), ("about", 500),
    ("who", 500), ("get", 450), ("which", 450), ("go", 450), ("me", 450),

    // Rank 51-100: very high frequency (400).
    ("when", 400), ("make", 400), ("can", 400), ("like", 400), ("time", 400),
    ("no", 400), ("just", 400), ("him", 400), ("know", 400), ("take", 400),
    ("people", 400), ("into", 400), ("year", 400), ("your", 400), ("good", 400),
    ("some", 400), ("could", 400), ("them", 400), ("see", 400), ("other", 400),
    ("than", 400), ("then", 400), ("now", 400), ("look", 400), ("only", 400),
    ("come", 400), ("its", 400), ("over", 400), ("think", 400), ("also", 400),
    ("back", 400), ("after", 400), ("use", 400), ("two", 400), ("how", 400),
    ("our", 400), ("work", 400), ("first", 400), ("well", 400), ("way", 400),
    ("even", 400), ("new", 400), ("want", 400), ("because", 400), ("any", 400),
    ("these", 400), ("give", 400), ("day", 400), ("most", 400), ("us", 400),

    // Rank 101-200: high frequency (300).
    ("find", 300), ("here", 300), ("thing", 300), ("many", 300), ("tell", 300),
    ("very", 300), ("should", 300), ("call", 300), ("still", 300), ("may", 300),
    ("own", 300), ("ask", 300), ("need", 300), ("try", 300), ("leave", 300),
    ("hand", 300), ("keep", 300), ("let", 300), ("put", 300), ("begin", 300),
    ("show", 300), ("part", 300), ("against", 300), ("place", 300), ("those", 300),
    ("between", 300), ("great", 300), ("again", 300), ("off", 300), ("before", 300),
    ("must", 300), ("through", 300), ("right", 300), ("too", 300), ("mean", 300),
    ("where", 300), ("help", 300), ("might", 300), ("much", 300), ("old", 300),
    ("each", 300), ("long", 300), ("world", 300), ("life", 300), ("since", 300),
    ("number", 300), ("same", 300), ("man", 300), ("both", 300), ("while", 300),
    ("little", 300), ("small", 300), ("more", 300), ("last", 300), ("such", 300),
    ("change", 300), ("point", 300), ("few", 300), ("end", 300), ("head", 300),
    ("high", 300), ("every", 300), ("run", 300), ("turn", 300), ("play", 300),
    ("real", 300), ("feel", 300), ("follow", 300), ("move", 300), ("seem", 300),
    ("start", 300), ("close", 300), ("night", 300), ("live", 300), ("under", 300),
    ("open", 300), ("believe", 300), ("country", 300), ("house", 300), ("school", 300),
    ("become", 300), ("never", 300), ("home", 300), ("water", 300), ("name", 300),
    ("state", 300), ("always", 300), ("why", 300), ("young", 300), ("read", 300),
    ("children", 300), ("side", 300), ("been", 300), ("boy", 300), ("group", 300),
    ("being", 300), ("large", 300), ("big", 300), ("above", 300), ("though", 300),

    // Rank 201-400: medium-high frequency (200).
    ("next", 200), ("set", 200), ("own", 200), ("stop", 200), ("face", 200),
    ("watch", 200), ("far", 200), ("walk", 200), ("girl", 200), ("miss", 200),
    ("stand", 200), ("sit", 200), ("write", 200), ("hear", 200), ("bring", 200),
    ("happen", 200), ("hold", 200), ("question", 200), ("light", 200), ("talk", 200),
    ("family", 200), ("body", 200), ("mind", 200), ("door", 200), ("car", 200),
    ("money", 200), ("story", 200), ("game", 200), ("power", 200), ("mother", 200),
    ("problem", 200), ("room", 200), ("city", 200), ("food", 200), ("father", 200),
    ("line", 200), ("idea", 200), ("friend", 200), ("second", 200), ("war", 200),
    ("book", 200), ("eye", 200), ("kind", 200), ("children", 200), ("land", 200),
    ("company", 200), ("system", 200), ("program", 200), ("woman", 200), ("pay", 200),
    ("lose", 200), ("meet", 200), ("learn", 200), ("word", 200), ("fact", 200),
    ("enough", 200), ("almost", 200), ("across", 200), ("during", 200), ("sure", 200),
    ("best", 200), ("better", 200), ("often", 200), ("among", 200), ("hard", 200),
    ("early", 200), ("along", 200), ("rather", 200), ("upon", 200), ("quite", 200),
    ("later", 200), ("less", 200), ("already", 200), ("until", 200), ("perhaps", 200),
    ("several", 200), ("however", 200), ("around", 200), ("without", 200), ("together", 200),
    ("case", 200), ("week", 200), ("nothing", 200), ("white", 200), ("bit", 200),
    ("black", 200), ("today", 200), ("course", 200), ("full", 200), ("possible", 200),
    ("able", 200), ("away", 200), ("half", 200), ("ago", 200), ("air", 200),
    ("area", 200), ("done", 200), ("important", 200), ("public", 200), ("report", 200),
    ("different", 200), ("service", 200), ("government", 200), ("general", 200), ("information", 200),
    ("business", 200), ("study", 200), ("going", 200), ("point", 200), ("order", 200),
    ("law", 200), ("level", 200), ("member", 200), ("political", 200), ("social", 200),
    ("national", 200), ("interest", 200), ("human", 200), ("president", 200), ("team", 200),
    ("market", 200), ("court", 200), ("force", 200), ("office", 200), ("music", 200),
    ("age", 200), ("sense", 200), ("top", 200), ("whole", 200), ("issue", 200),
    ("care", 200), ("cup", 200), ("short", 200), ("fire", 200), ("strong", 200),
    ("free", 200), ("clear", 200), ("break", 200), ("deal", 200), ("death", 200),
    ("pass", 200), ("south", 200), ("west", 200), ("north", 200), ("east", 200),
    ("common", 200), ("fall", 200), ("reason", 200), ("field", 200), ("rate", 200),
    ("cost", 200), ("result", 200), ("action", 200), ("job", 200), ("local", 200),
    ("plan", 200), ("return", 200), ("support", 200), ("control", 200), ("class", 200),
    ("certain", 200), ("matter", 200), ("per", 200), ("role", 200), ("five", 200),
    ("three", 200), ("four", 200), ("six", 200), ("seven", 200), ("eight", 200),
    ("nine", 200), ("ten", 200), ("hundred", 200), ("thousand", 200), ("million", 200),

    // Rank 401-700: medium frequency (100).
    ("cut", 100), ("red", 100), ("hit", 100), ("record", 100), ("hot", 100),
    ("love", 100), ("war", 100), ("church", 100), ("kill", 100), ("wife", 100),
    ("husband", 100), ("daughter", 100), ("son", 100), ("sister", 100), ("brother", 100),
    ("morning", 100), ("evening", 100), ("table", 100), ("sit", 100), ("bed", 100),
    ("window", 100), ("foot", 100), ("street", 100), ("road", 100), ("town", 100),
    ("party", 100), ("ground", 100), ("nature", 100), ("future", 100), ("drive", 100),
    ("fight", 100), ("nation", 100), ("space", 100), ("peace", 100), ("god", 100),
    ("building", 100), ("include", 100), ("remain", 100), ("continue", 100), ("form", 100),
    ("picture", 100), ("increase", 100), ("create", 100), ("accept", 100), ("provide", 100),
    ("receive", 100), ("allow", 100), ("lead", 100), ("spend", 100), ("produce", 100),
    ("expect", 100), ("serve", 100), ("die", 100), ("raise", 100), ("reach", 100),
    ("grow", 100), ("send", 100), ("remember", 100), ("carry", 100), ("wait", 100),
    ("require", 100), ("consider", 100), ("offer", 100), ("buy", 100), ("sell", 100),
    ("build", 100), ("choose", 100), ("develop", 100), ("wish", 100), ("sign", 100),
    ("fill", 100), ("draw", 100), ("rest", 100), ("eat", 100), ("wear", 100),
    ("agree", 100), ("appear", 100), ("note", 100), ("catch", 100), ("pick", 100),
    ("decide", 100), ("drop", 100), ("win", 100), ("cause", 100), ("hope", 100),
    ("pull", 100), ("push", 100), ("throw", 100), ("share", 100), ("step", 100),
    ("add", 100), ("develop", 100), ("rise", 100), ("spoke", 100), ("press", 100),
    ("describe", 100), ("sound", 100), ("act", 100), ("touch", 100), ("report", 100),
    ("mark", 100), ("save", 100), ("test", 100), ("cover", 100), ("center", 100),
    ("value", 100), ("model", 100), ("effect", 100), ("trade", 100), ("land", 100),
    ("clear", 100), ("paper", 100), ("view", 100), ("effort", 100), ("special", 100),
    ("major", 100), ("economic", 100), ("simple", 100), ("personal", 100), ("military", 100),
    ("community", 100), ("heart", 100), ("blood", 100), ("bank", 100), ("position", 100),
    ("period", 100), ("similar", 100), ("single", 100), ("current", 100), ("likely", 100),
    ("natural", 100), ("final", 100), ("recent", 100), ("central", 100), ("particular", 100),
    ("director", 100), ("whether", 100), ("whose", 100), ("whom", 100), ("either", 100),
    ("itself", 100), ("himself", 100), ("herself", 100), ("myself", 100), ("themselves", 100),
    ("everything", 100), ("everyone", 100), ("something", 100), ("anything", 100), ("someone", 100),
    ("sometimes", 100), ("perhaps", 100), ("especially", 100), ("certainly", 100), ("actually", 100),
    ("probably", 100), ("usually", 100), ("simply", 100), ("quickly", 100), ("suddenly", 100),
    ("america", 100), ("american", 100), ("united", 100), ("states", 100), ("english", 100),
    ("human", 100), ("history", 100), ("century", 100), ("modern", 100), ("society", 100),
    ("education", 100), ("technology", 100), ("computer", 100), ("science", 100), ("health", 100),
    ("college", 100), ("student", 100), ("teacher", 100), ("doctor", 100), ("hospital", 100),
    ("tonight", 100), ("today", 100), ("tomorrow", 100), ("yesterday", 100), ("monday", 100),
    ("tuesday", 100), ("wednesday", 100), ("thursday", 100), ("friday", 100), ("saturday", 100),
    ("sunday", 100), ("january", 100), ("february", 100), ("march", 100), ("april", 100),
    ("june", 100), ("july", 100), ("august", 100), ("september", 100), ("october", 100),
    ("november", 100), ("december", 100), ("spring", 100), ("summer", 100), ("winter", 100),

    // Rank 701-1000: moderate frequency (50).
    ("shall", 50), ("dear", 50), ("glad", 50), ("fear", 50), ("ring", 50),
    ("sail", 50), ("fly", 50), ("snow", 50), ("rain", 50), ("cold", 50),
    ("warm", 50), ("deep", 50), ("wide", 50), ("dark", 50), ("bright", 50),
    ("soft", 50), ("sweet", 50), ("rough", 50), ("smooth", 50), ("sharp", 50),
    ("thick", 50), ("thin", 50), ("flat", 50), ("round", 50), ("quick", 50),
    ("slow", 50), ("safe", 50), ("rich", 50), ("poor", 50), ("wild", 50),
    ("true", 50), ("false", 50), ("wrong", 50), ("strange", 50), ("beautiful", 50),
    ("wonder", 50), ("explain", 50), ("suggest", 50), ("supply", 50), ("demand", 50),
    ("attack", 50), ("defend", 50), ("destroy", 50), ("protect", 50), ("measure", 50),
    ("gather", 50), ("spread", 50), ("travel", 50), ("search", 50), ("discover", 50),
    ("express", 50), ("prevent", 50), ("contain", 50), ("compare", 50), ("remove", 50),
    ("replace", 50), ("apply", 50), ("improve", 50), ("perform", 50), ("claim", 50),
    ("reduce", 50), ("establish", 50), ("challenge", 50), ("feature", 50), ("quality", 50),
    ("trouble", 50), ("opportunity", 50), ("condition", 50), ("situation", 50), ("according", 50),
    ("statement", 50), ("security", 50), ("authority", 50), ("industry", 50), ("official", 50),
    ("operation", 50), ("movement", 50), ("evidence", 50), ("management", 50), ("material", 50),
    ("individual", 50), ("practice", 50), ("traditional", 50), ("significant", 50), ("standard", 50),
    ("determine", 50), ("performance", 50), ("development", 50), ("experience", 50), ("knowledge", 50),
    ("international", 50), ("population", 50), ("production", 50), ("environment", 50), ("research", 50),
    ("relationship", 50), ("professional", 50), ("organization", 50), ("available", 50), ("financial", 50),

    // Rank 1001-1500: lower-medium frequency (30).
    ("bear", 30), ("dare", 30), ("shake", 30), ("hang", 30), ("lay", 30),
    ("blow", 30), ("stick", 30), ("fix", 30), ("sing", 30), ("shut", 30),
    ("rain", 30), ("hide", 30), ("lock", 30), ("jump", 30), ("smoke", 30),
    ("smile", 30), ("dance", 30), ("swim", 30), ("climb", 30), ("cook", 30),
    ("wash", 30), ("dress", 30), ("brush", 30), ("clean", 30), ("pour", 30),
    ("wrap", 30), ("lift", 30), ("tear", 30), ("fold", 30), ("bend", 30),
    ("roll", 30), ("mix", 30), ("crash", 30), ("slip", 30), ("land", 30),
    ("dust", 30), ("chain", 30), ("iron", 30), ("blade", 30), ("bone", 30),
    ("brain", 30), ("chest", 30), ("knee", 30), ("neck", 30), ("shoulder", 30),
    ("lip", 30), ("nose", 30), ("ear", 30), ("tooth", 30), ("tongue", 30),
    ("finger", 30), ("stomach", 30), ("palm", 30), ("thumb", 30), ("skin", 30),
    ("rice", 30), ("bread", 30), ("meat", 30), ("fruit", 30), ("milk", 30),
    ("sugar", 30), ("salt", 30), ("cheese", 30), ("butter", 30), ("chicken", 30),
    ("fish", 30), ("apple", 30), ("orange", 30), ("potato", 30), ("coffee", 30),
    ("cat", 30), ("dog", 30), ("horse", 30), ("bird", 30), ("tree", 30),
    ("flower", 30), ("grass", 30), ("wood", 30), ("rock", 30), ("stone", 30),
    ("hill", 30), ("lake", 30), ("river", 30), ("sea", 30), ("sky", 30),
    ("cloud", 30), ("moon", 30), ("sun", 30), ("star", 30), ("earth", 30),
    ("island", 30), ("mountain", 30), ("valley", 30), ("beach", 30), ("forest", 30),
    ("ocean", 30), ("desert", 30), ("garden", 30), ("bridge", 30), ("farm", 30),
    ("village", 30), ("clothes", 30), ("shirt", 30), ("shoe", 30), ("hat", 30),
    ("coat", 30), ("bag", 30), ("ball", 30), ("box", 30), ("key", 30),
    ("cup", 30), ("chair", 30), ("plate", 30), ("bottle", 30), ("tool", 30),
    ("gun", 30), ("knife", 30), ("oil", 30), ("gas", 30), ("steel", 30),
    ("glass", 30), ("stone", 30), ("wire", 30), ("rope", 30), ("gold", 30),
    ("silver", 30), ("flag", 30), ("map", 30), ("ticket", 30), ("clock", 30),
    ("dollar", 30), ("mark", 30), ("mile", 30), ("inch", 30), ("yard", 30),
    ("weight", 30), ("shape", 30), ("edge", 30), ("corner", 30), ("bottom", 30),
    ("surface", 30), ("hole", 30), ("spot", 30), ("path", 30), ("track", 30),
    ("station", 30), ("bus", 30), ("ship", 30), ("boat", 30), ("plane", 30),
    ("train", 30), ("truck", 30), ("wheel", 30), ("wing", 30), ("engine", 30),

    // Rank 1501-2500: low-medium frequency (15).
    ("crew", 15), ("dock", 15), ("bay", 15), ("sail", 15), ("harbor", 15),
    ("cabin", 15), ("deck", 15), ("mast", 15), ("storm", 15), ("wave", 15),
    ("shore", 15), ("beach", 15), ("cliff", 15), ("cave", 15), ("stream", 15),
    ("pool", 15), ("pond", 15), ("marsh", 15), ("trail", 15), ("fence", 15),
    ("gate", 15), ("barn", 15), ("tower", 15), ("castle", 15), ("palace", 15),
    ("temple", 15), ("camp", 15), ("tent", 15), ("cabin", 15), ("hall", 15),
    ("garage", 15), ("roof", 15), ("wall", 15), ("floor", 15), ("ceiling", 15),
    ("porch", 15), ("stair", 15), ("shelf", 15), ("drawer", 15), ("closet", 15),
    ("pipe", 15), ("tank", 15), ("pump", 15), ("wire", 15), ("switch", 15),
    ("plug", 15), ("cord", 15), ("panel", 15), ("frame", 15), ("screen", 15),
    ("lens", 15), ("mirror", 15), ("chart", 15), ("desk", 15), ("bench", 15),
    ("counter", 15), ("carpet", 15), ("curtain", 15), ("blanket", 15), ("pillow", 15),
    ("towel", 15), ("basket", 15), ("bucket", 15), ("brush", 15), ("comb", 15),
    ("pin", 15), ("needle", 15), ("thread", 15), ("cloth", 15), ("silk", 15),
    ("cotton", 15), ("leather", 15), ("rubber", 15), ("plastic", 15), ("metal", 15),
    ("copper", 15), ("tin", 15), ("lead", 15), ("coal", 15), ("fuel", 15),
    ("acid", 15), ("powder", 15), ("grain", 15), ("seed", 15), ("root", 15),
    ("leaf", 15), ("branch", 15), ("stem", 15), ("bush", 15), ("weed", 15),
    ("crop", 15), ("wheat", 15), ("corn", 15), ("cotton", 15), ("tobacco", 15),
    ("soil", 15), ("mud", 15), ("clay", 15), ("sand", 15), ("frost", 15),
    ("fog", 15), ("dew", 15), ("flood", 15), ("earthquake", 15), ("volcano", 15),
    ("thunder", 15), ("lightning", 15), ("sunset", 15), ("dawn", 15), ("dusk", 15),
    ("shadow", 15), ("flame", 15), ("spark", 15), ("ash", 15), ("smoke", 15),
    ("heat", 15), ("chill", 15), ("shade", 15), ("glow", 15), ("beam", 15),
    ("ray", 15), ("spectrum", 15), ("color", 15), ("shade", 15), ("tone", 15),
    ("music", 15), ("song", 15), ("poem", 15), ("verse", 15), ("drama", 15),
    ("novel", 15), ("tale", 15), ("myth", 15), ("legend", 15), ("hero", 15),
    ("villain", 15), ("knight", 15), ("prince", 15), ("princess", 15), ("queen", 15),
    ("king", 15), ("lord", 15), ("master", 15), ("slave", 15), ("priest", 15),
    ("angel", 15), ("devil", 15), ("soul", 15), ("ghost", 15), ("spirit", 15),
    ("wish", 15), ("dream", 15), ("sleep", 15), ("wake", 15), ("born", 15),
    ("die", 15), ("marry", 15), ("pray", 15), ("cry", 15), ("laugh", 15),
    ("anger", 15), ("joy", 15), ("pride", 15), ("shame", 15), ("guilt", 15),
    ("pain", 15), ("pleasure", 15), ("comfort", 15), ("hunger", 15), ("thirst", 15),
    ("courage", 15), ("danger", 15), ("risk", 15), ("luck", 15), ("fate", 15),
    ("honor", 15), ("duty", 15), ("justice", 15), ("mercy", 15), ("truth", 15),
    ("wisdom", 15), ("virtue", 15), ("grace", 15), ("beauty", 15), ("charm", 15),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arpabet_hello() {
        let ipa = arpabet_to_ipa("HH AH0 L OW1");
        assert_eq!(ipa, vec!["h", "ə", "l", "oʊ"]);
    }

    #[test]
    fn arpabet_delete() {
        let ipa = arpabet_to_ipa("D IH0 L IY1 T");
        assert_eq!(ipa, vec!["d", "ɪ", "l", "iː", "t"]);
    }

    #[test]
    fn arpabet_cat() {
        let ipa = arpabet_to_ipa("K AE1 T");
        assert_eq!(ipa, vec!["k", "æ", "t"]);
    }

    #[test]
    fn arpabet_think() {
        let ipa = arpabet_to_ipa("TH IH1 NG K");
        assert_eq!(ipa, vec!["θ", "ɪ", "ŋ", "k"]);
    }

    #[test]
    fn arpabet_church() {
        let ipa = arpabet_to_ipa("CH ER1 CH");
        assert_eq!(ipa, vec!["tʃ", "ɜː", "tʃ"]);
    }

    #[test]
    fn arpabet_schwa_reduction() {
        // Stressed AH = ʌ, unstressed AH = ə.
        let ipa = arpabet_to_ipa("AH1 B AH0 V");
        assert_eq!(ipa, vec!["ʌ", "b", "ə", "v"]);
    }

    #[test]
    fn load_cmudict_if_available() {
        let path = Path::new("data/cmudict.dict");
        if path.exists() {
            let trie = load_cmudict(path).expect("should load cmudict");
            assert!(
                trie.len() > 100_000,
                "expected 100K+ words, got {}",
                trie.len()
            );

            // Verify a known word — "hello" = HH AH0 L OW1 = "həloʊ".
            let phonemes = crate::pronunciation::ipa_to_phoneme_sequence("həloʊ");
            let results = trie.lookup_exact(&phonemes);
            assert!(
                !results.is_empty(),
                "should find 'hello' in trie, phonemes={:?}",
                phonemes
            );
        }
    }

    #[test]
    fn trie_lookup_common_words() {
        let path = Path::new("data/cmudict.dict");
        if !path.exists() {
            return;
        }
        let trie = load_cmudict(path).expect("should load cmudict");

        // Test several common words.
        let test_cases = [
            ("kæt", "cat"),
            ("dɔɡ", "dog"),
            ("ðə", "the"),
            ("ʌv", "of"),
            ("ænd", "and"),
        ];
        for (ipa, expected_word) in &test_cases {
            let phonemes = crate::pronunciation::ipa_to_phoneme_sequence(ipa);
            let results = trie.lookup_exact(&phonemes);
            let words: Vec<&str> = results.iter().map(|r| r.word.as_str()).collect();
            assert!(
                words.contains(expected_word),
                "Expected '{}' for /{}/. Phonemes={:?}, Got={:?}",
                expected_word, ipa, phonemes, words
            );
        }
    }

    #[test]
    fn fuzzy_lookup_finds_close_matches() {
        let path = Path::new("data/cmudict.dict");
        if !path.exists() {
            return;
        }
        let trie = load_cmudict(path).expect("should load cmudict");

        // Fuzzy lookup for "cat" /kæt/ — should also find "bat", "hat", etc.
        let phonemes = crate::pronunciation::ipa_to_phoneme_sequence("kæt");
        let results = trie.lookup_fuzzy(&phonemes, 1);
        let words: Vec<&str> = results.iter().map(|r| r.word.as_str()).collect();
        assert!(words.contains(&"cat"), "exact match should appear");
        // At least some fuzzy matches.
        assert!(
            results.len() > 1,
            "should find fuzzy matches, got {}",
            results.len()
        );
    }
}
