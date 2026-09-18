package com.zhou.craft.controller;

import com.zhou.craft.model.CraftWork;
import com.zhou.craft.repo.CraftWorkRepository;
import org.springframework.web.bind.annotation.*;
import java.util.*;

@RestController
@RequestMapping("/api/works")
public class WorkController {
    private final CraftWorkRepository repo;
    public WorkController(CraftWorkRepository repo){this.repo=repo;}

    @GetMapping
    public List<CraftWork> list(@RequestParam(required=false) String category,@RequestParam(required=false) String keyword){
        List<CraftWork> data = category==null || category.isBlank() ? repo.findAll() : repo.findByCategoryIgnoreCase(category);
        if(keyword==null || keyword.isBlank()) return data;
        String k=keyword.toLowerCase();
        return data.stream().filter(w->w.getTitle().toLowerCase().contains(k)||w.getSummary().toLowerCase().contains(k)||w.getCreatorName().toLowerCase().contains(k)).toList();
    }

    @GetMapping("/featured")
    public List<CraftWork> featured(){return repo.findByFeaturedTrueOrderByLikesDesc();}

    @GetMapping("/{id}")
    public CraftWork detail(@PathVariable Long id){return repo.findById(id).orElseThrow();}
}
