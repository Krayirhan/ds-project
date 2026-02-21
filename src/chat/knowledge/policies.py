from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    category: str
    tags: list[str]
    title: str
    content: str
    priority: int = 5


KNOWLEDGE_BASE: list[KnowledgeChunk] = [
    # ── Depozito politikaları ────────────────────────────────────────────────
    KnowledgeChunk(
        chunk_id="pol_001",
        category="cancellation_policy",
        tags=["depozito", "non_refund", "iptal", "iade"],
        title="İade edilmez depozito — yüksek risk profili",
        content=(
            "Dikkat: Gerçek rezervasyon verilerinde iade edilmez depozitolu müşterilerin iptal oranı "
            "beklenenden çok daha yüksektir. Bu profil zaten riskli olarak değerlendirilip "
            "iade edilmez depozitoya yönlendirilmiş olabilir. "
            "Depozito tek başına yeterli güvence değildir; model skoru geçerlidir. "
            "Müşteriye kişisel teyit mesajı gönderin, oda hazırlığını önceden onaylayın "
            "ve tarih değişikliğini finansal kaybın alternatifi olarak net şekilde anlatın."
        ),
        priority=1,
    ),
    KnowledgeChunk(
        chunk_id="pol_002",
        category="cancellation_policy",
        tags=["depozito", "no_deposit", "iptal", "yüksek_risk"],
        title="Depozitosuz rezervasyon",
        content=(
            "Depozitosuz rezervasyonlarda müşteri finansal bağlılık hissetmez; iptal maliyeti sıfırdır. "
            "Küçük bir avantaj (ücretsiz kahvaltı, erken giriş vb.) karşılığında ön ödeme talep etmek "
            "iptali %20-30 oranında azaltır. Depozito politikasını nazikçe hatırlatın."
        ),
        priority=1,
    ),
    KnowledgeChunk(
        chunk_id="pol_014",
        category="cancellation_policy",
        tags=["refundable", "iade", "iptal", "orta_risk"],
        title="İade edilir depozito politikası",
        content=(
            "İade edilebilir politikada müşteri check-in'den 48 saat öncesine kadar "
            "ücretsiz iptal hakkına sahiptir. Bu müşteriye check-in tarihine 3 gün kala "
            "kısa bir hatırlatma yapın; son 48 saatte iptal riski dramatik düşer. "
            "Kapanış penceresi yaklaştıkça müşteriyi bilgilendirmek bağlılığı artırır."
        ),
        priority=2,
    ),
    # ── Zaman ve lead-time politikaları ─────────────────────────────────────
    KnowledgeChunk(
        chunk_id="pol_003",
        category="retention",
        tags=["lead_time", "uzun", "erken_rezervasyon", "yüksek_risk"],
        title="Uzun süre önce yapılan rezervasyon",
        content=(
            "180 günden fazla önce yapılan rezervasyonlarda plan değişikliği riski yüksektir. "
            "Check-in tarihine yaklaşırken düzenli hatırlatma ve esnek tarih değişikliği önerin. "
            "60 gün, 30 gün ve 7 gün öncesinde otomatik hatırlatma gönderilmesi önerilir."
        ),
        priority=2,
    ),
    KnowledgeChunk(
        chunk_id="pol_007",
        category="retention",
        tags=["check_in", "yakın", "son_dakika", "orta_risk"],
        title="Check-in tarihine yakın rezervasyon",
        content=(
            "Son 7 gün içinde yapılan rezervasyonlarda iptal oranı belirgin düşüktür; "
            "müşteri genellikle hazır ve kararlıdır. "
            "Hızlı bir karşılama mesajı göndererek bağlılığı pekiştirin. "
            "Oda hazırlığı ve özel istekleri önceden teyit edin."
        ),
        priority=3,
    ),
    # ── Müşteri profili ──────────────────────────────────────────────────────
    KnowledgeChunk(
        chunk_id="pol_004",
        category="retention",
        tags=["geçmiş_iptal", "previous_cancellations", "yüksek_risk"],
        title="Geçmişte iptal etmiş müşteri",
        content=(
            "Daha önce iptal etmiş müşteri yüksek riskli profildir. "
            "24 saat içinde kısa ve kişisel bir teyit araması yapın. "
            "Depozito talep etmek veya küçük bir avantaj sunmak rezervasyonu koruma şansını artırır. "
            "Müşteri adını kullanmak ve özel isteklerine değinmek bağlılığı güçlendirir."
        ),
        priority=1,
    ),
    KnowledgeChunk(
        chunk_id="pol_009",
        category="retention",
        tags=["sadık", "tekrar", "düşük_risk", "upsell"],
        title="Sadık müşteri yönetimi",
        content=(
            "Daha önce konaklama yapmış tekrarlayan müşteri iptal riskine karşı en güçlü "
            "profildir; bağlılık oranı ortalama %85'in üzerindedir. "
            "Bu müşteriye sadakat avantajı, sürpriz karşılama veya kişisel not sunun. "
            "Bağlılığı güçlendirmek aynı zamanda gelecek rezervasyon olasılığını da artırır."
        ),
        priority=2,
    ),
    # ── Kanal ve segment ────────────────────────────────────────────────────
    KnowledgeChunk(
        chunk_id="pol_005",
        category="segment",
        tags=["online", "ota", "online_ta", "yüksek_risk"],
        title="Online acente (OTA) rezervasyonu",
        content=(
            "Online acente kanalında müşteri kolayca rakip fiyat karşılaştırması yapabilir. "
            "Otel ile doğrudan iletişim kurarak net değer önerisi sunmak iptal riskini azaltır. "
            "Doğrudan kanal avantajlarını (esnek iptal, oda yükseltme, sadakat puanı) vurgulayın. "
            "Online TA aracılığı gelen rezervasyonlarda depozito talebi özellikle etkilidir."
        ),
        priority=2,
    ),
    KnowledgeChunk(
        chunk_id="pol_008",
        category="segment",
        tags=["corporate", "kurumsal", "direkt", "düşük_risk"],
        title="Kurumsal kanal müşterisi",
        content=(
            "Kurumsal rezervasyonlarda şirket seyahat politikası geçerlidir; "
            "iptal kararı bireysel değil kurumsal düzeyde alınır. "
            "Şirket hesabını oluşturun, kontak kişiyi ve onay sürecini teyit edin. "
            "Kurumsal müşterilerde iptal genellikle operasyonel nedenlerle gerçekleşir; "
            "tarih değişikliği alternatifi sunmak rezervasyonu kurtarabilir."
        ),
        priority=4,
    ),
    KnowledgeChunk(
        chunk_id="pol_013",
        category="retention",
        tags=["grup", "çok_kişi", "yüksek_risk"],
        title="Grup rezervasyonu riski",
        content=(
            "2'den fazla oda veya 5+ kişilik grup rezervasyonlarda tek bir iptal "
            "diğerlerini tetikleyebilir; domino etkisi riski yüksektir. "
            "Grup liderini belirleyin, grup depozito politikasını uygulayın "
            "ve iptal koşullarını yazılı olarak karşılıklı teyit ettirin."
        ),
        priority=2,
    ),
    # ── Aksiyon ve iletişim ──────────────────────────────────────────────────
    KnowledgeChunk(
        chunk_id="pol_011",
        category="communication",
        tags=["iletişim", "arama", "teyit", "proaktif"],
        title="Proaktif müşteri iletişimi",
        content=(
            "Check-in tarihine 48-72 saat kala kişisel bir teyit araması "
            "%18-22 oranında iptal riskini azaltır. "
            "Kısa ve samimi bir mesaj, müşteri adı kullanımı ve özel isteklere değinmek "
            "bağ kurmayı güçlendirir. SMS veya WhatsApp üzerinden gönderilen kişisel "
            "karşılama mesajları da oldukça etkilidir."
        ),
        priority=1,
    ),
    KnowledgeChunk(
        chunk_id="pol_010",
        category="retention",
        tags=["tarih_değişikliği", "esnek", "öneri"],
        title="Tarih değişikliği seçeneği",
        content=(
            "İptal yerine tarih değişikliği teklif etmek rezervasyonu kurtarmanın "
            "en etkili ve düşük maliyetli yöntemidir. "
            "Esnek değişim politikası sunan otel, müşteriyi elde tutar ve "
            "doluluk planlamasını da kolaylaştırır. "
            "Tarih değişikliğinde ek ücret almamak müşteriyi kazanmanın anahtarıdır."
        ),
        priority=2,
    ),
    KnowledgeChunk(
        chunk_id="pol_015",
        category="retention",
        tags=["yüksek_risk", "acil", "aksiyon"],
        title="Yüksek riskli durum acil protokolü",
        content=(
            "Risk %65'in üzerindeyse acil müdahale protokolü devreye girer: "
            "1) Bugün kişisel teyit araması yap. "
            "2) Küçük bir avantaj teklif et (ücretsiz kahvaltı, erken giriş vb.). "
            "3) Depozito politikasını nazikçe hatırlat. "
            "4) İptal yerine tarih değişikliği seçeneği sun. "
            "Dört adımın tamamı uygulandığında iptal riski %35-40 düşer."
        ),
        priority=1,
    ),
    # ── Upsell ve gelir artırma ──────────────────────────────────────────────
    KnowledgeChunk(
        chunk_id="pol_006",
        category="upsell",
        tags=["düşük_risk", "upsell", "ek_hizmet"],
        title="Düşük riskte ek gelir fırsatı",
        content=(
            "Düşük riskli müşteriye şu ek hizmetler önerilebilir: "
            "oda yükseltme, havalimanı transfer, spa paketi, paket kahvaltı, "
            "honeymoon veya doğum günü kutlama paketi. "
            "Upsell başarısı ortalama %15-25 ek gelir sağlar. "
            "Bu yaklaşım hem geliri artırır hem de müşteri bağlılığını güçlendirir."
        ),
        priority=3,
    ),
    KnowledgeChunk(
        chunk_id="pol_012",
        category="upsell",
        tags=["upsell", "ek_hizmet", "paket", "sadık"],
        title="Yüksek değerli upsell paketleri",
        content=(
            "Sadık veya düşük riskli müşterilere özel paketler: "
            "erken check-in / geç check-out (ücretsiz veya sembolik ücretle), "
            "şarap / meyve sepeti karşılama, VIP havalimanı transfer, "
            "tam pansiyon geçişi, aktivite paketi (plaj kabuğu, snorkel, bisiklet). "
            "Paketin değerini müşterinin profiliyle eşleştirmek kabul oranını artırır."
        ),
        priority=3,
    ),
]
