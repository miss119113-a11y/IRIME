package com.zhou.craft.repo;
import com.zhou.craft.model.WorkLike;import org.springframework.data.jpa.repository.JpaRepository;import java.util.*;
public interface WorkLikeRepository extends JpaRepository<WorkLike,Long>{Optional<WorkLike> findByUserIdAndWorkId(Long userId,Long workId);List<WorkLike> findByUserId(Long userId);}