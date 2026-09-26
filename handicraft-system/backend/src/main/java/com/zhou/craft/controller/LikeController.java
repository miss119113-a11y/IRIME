package com.zhou.craft.controller;
import com.zhou.craft.model.*;import com.zhou.craft.repo.*;import com.zhou.craft.security.AuthContext;import org.springframework.web.bind.annotation.*;import java.util.*;
@RestController @RequestMapping("/api/likes")
public class LikeController{
private final WorkLikeRepository likes;private final CraftWorkRepository works;
public LikeController(WorkLikeRepository likes,CraftWorkRepository works){this.likes=likes;this.works=works;}
@GetMapping("/work/{id}")public Map<String,Object> status(@PathVariable Long id){Long uid=AuthContext.require().id();var w=works.findById(id).orElseThrow();return Map.of("liked",likes.findByUserIdAndWorkId(uid,id).isPresent(),"likes",w.getLikes());}
@PostMapping("/work/{id}")public Map<String,Object> toggle(@PathVariable Long id){Long uid=AuthContext.require().id();var w=works.findById(id).orElseThrow();var current=likes.findByUserIdAndWorkId(uid,id);boolean liked;if(current.isPresent()){likes.delete(current.get());w.removeLike();liked=false;}else{likes.save(new WorkLike(uid,id));w.addLike();liked=true;}works.save(w);return Map.of("liked",liked,"likes",w.getLikes());}
}