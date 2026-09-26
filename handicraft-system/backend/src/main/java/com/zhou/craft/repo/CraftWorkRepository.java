package com.zhou.craft.repo;
import com.zhou.craft.model.CraftWork;import org.springframework.data.jpa.repository.JpaRepository;import java.util.*;
public interface CraftWorkRepository extends JpaRepository<CraftWork,Long>{
List<CraftWork> findByFeaturedTrueOrderByLikesDesc();List<CraftWork> findByCategoryIgnoreCase(String category);Optional<CraftWork> findByTitle(String title);
}