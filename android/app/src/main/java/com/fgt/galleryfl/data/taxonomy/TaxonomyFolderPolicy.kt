package com.fgt.galleryfl.data.taxonomy

data class TaxonomyCategory(
    val id: String,
    val name: String,
    val children: List<TaxonomyTag>
)

data class TaxonomyTag(
    val id: String,
    val name: String,
    val defaultThreshold: Float = 0.5f
)

data class FolderPolicy(
    val category: TaxonomyCategory,
    val tag: TaxonomyTag,
    val minImagesToCreateFolder: Int = 3
)
