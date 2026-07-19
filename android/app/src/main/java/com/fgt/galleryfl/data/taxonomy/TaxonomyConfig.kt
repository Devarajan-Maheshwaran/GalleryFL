package com.fgt.galleryfl.data.taxonomy

/**
 * Detailed gallery taxonomy plus the seven parent outputs of the ML head.
 *
 * The 34 leaf tags remain product metadata for future sub-tag personalization;
 * they are not output neurons. Model indices always follow [modelTags].
 */
object TaxonomyConfig {

    val categories = listOf(
        TaxonomyCategory("people", "People", listOf(
            TaxonomyTag("selfie", "Selfie", 0.6f),
            TaxonomyTag("group_photo", "Group Photo", 0.5f),
            TaxonomyTag("portrait", "Portrait", 0.5f),
            TaxonomyTag("crowd", "Crowd", 0.5f)
        )),
        TaxonomyCategory("places", "Places", listOf(
            TaxonomyTag("beach", "Beach", 0.5f),
            TaxonomyTag("mountain", "Mountain", 0.5f),
            TaxonomyTag("city", "City", 0.5f),
            TaxonomyTag("indoor", "Indoor", 0.5f),
            TaxonomyTag("rural", "Rural", 0.5f)
        )),
        TaxonomyCategory("activities", "Activities", listOf(
            TaxonomyTag("sports", "Sports", 0.6f),
            TaxonomyTag("cooking", "Cooking", 0.5f),
            TaxonomyTag("celebration", "Celebration", 0.5f),
            TaxonomyTag("work", "Work", 0.5f),
            TaxonomyTag("travel", "Travel", 0.5f)
        )),
        TaxonomyCategory("objects", "Objects", listOf(
            TaxonomyTag("food", "Food", 0.6f),
            TaxonomyTag("vehicle", "Vehicle", 0.5f),
            TaxonomyTag("gadget", "Gadget", 0.5f),
            TaxonomyTag("clothing", "Clothing", 0.5f),
            TaxonomyTag("art", "Art", 0.5f)
        )),
        TaxonomyCategory("documents", "Documents", listOf(
            TaxonomyTag("screenshot", "Screenshot", 0.7f),
            TaxonomyTag("receipt", "Receipt", 0.7f),
            TaxonomyTag("id_card", "ID Card", 0.8f),
            TaxonomyTag("handwritten", "Handwritten", 0.6f),
            TaxonomyTag("printed", "Printed", 0.6f)
        )),
        TaxonomyCategory("nature", "Nature", listOf(
            TaxonomyTag("landscape", "Landscape", 0.5f),
            TaxonomyTag("animal", "Animal", 0.6f),
            TaxonomyTag("plant", "Plant", 0.5f),
            TaxonomyTag("sky", "Sky", 0.5f),
            TaxonomyTag("water", "Water", 0.5f)
        )),
        TaxonomyCategory("events", "Events", listOf(
            TaxonomyTag("wedding", "Wedding", 0.6f),
            TaxonomyTag("birthday", "Birthday", 0.6f),
            TaxonomyTag("concert", "Concert", 0.6f),
            TaxonomyTag("graduation", "Graduation", 0.6f),
            TaxonomyTag("holiday", "Holiday", 0.5f)
        ))
    )

    /** Detailed tags are metadata, not model outputs. */
    val leafTags: List<TaxonomyTag> = categories.flatMap { it.children }

    /** Exact order shared with server/retrain/config.py and model_schema.json. */
    val modelTags: List<TaxonomyTag> = categories.map {
        TaxonomyTag(it.id, it.name, 0.0f)
    }

    const val NUM_CLASSES: Int = 7

    init {
        check(modelTags.size == NUM_CLASSES) { "GalleryFL model taxonomy must have seven parents" }
    }

    fun getPolicyForClassIndex(index: Int): FolderPolicy? {
        if (index !in modelTags.indices) return null
        return FolderPolicy(categories[index], modelTags[index])
    }
}
