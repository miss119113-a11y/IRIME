package com.zhou.craft.controller;
import com.zhou.craft.dto.ReviewForm;import com.zhou.craft.model.WorkReview;import com.zhou.craft.repo.*;import com.zhou.craft.security.AuthContext;import jakarta.validation.Valid;import org.springframework.web.bind.annotation.*;import java.util.*;
@RestController @RequestMapping("/api/reviews")public class ReviewController{
private final WorkReviewRepository reviews;private final CraftWorkRepository works;public ReviewController(WorkReviewRepository r,CraftWorkRepository w){reviews=r;works=w;}
@GetMapping("/work/{id}")public List<WorkReview> list(@PathVariable Long id){return reviews.findByWorkIdOrderByCreatedAtDesc(id);}
@PostMapping("/work/{id}")public WorkReview save(@PathVariable Long id,@Valid @RequestBody ReviewForm f){works.findById(id).orElseThrow();var p=AuthContext.require();var r=reviews.findByUserIdAndWorkId(p.id(),id).orElse(new WorkReview(p.id(),p.displayName(),id,f.rating(),f.content()));r.update(f.rating(),f.content());return reviews.save(r);}
}