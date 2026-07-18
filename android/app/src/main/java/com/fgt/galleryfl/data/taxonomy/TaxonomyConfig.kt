package com.fgt.galleryfl.data.taxonomy

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

    // Deterministic ordering of leaf tags for the classification head
    val leafTags: List<TaxonomyTag> = categories.flatMap { it.children }
    
    // Total number of classes
    val NUM_CLASSES = leafTags.size

    // Map class index to folder policy
    fun getPolicyForClassIndex(index: Int): FolderPolicy? {
        if (index < 0 || index >= leafTags.size) return null
        val tag = leafTags[index]
        val category = categories.find { it.children.contains(tag) } ?: return null
        // Avoid creating a noisy album from a single uncertain prediction.
        // FolderPolicy's default keeps the organisation preview and execution
        // threshold aligned at three matching images.
        return FolderPolicy(category, tag)
    }
}
