"""Prompt templates and functions for STS2 GRPO."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from sts2_env.cards.ironclad_cards_dict import ironclad_cards
except ImportError:
    ironclad_cards = {}


SYSTEM_PROMPT_BASE = """
You are playing Slay the Spire 2. Make optimal decisions to win the combat.

Your task is to analyze the current state, including:
- player: hp, block, energy, hand card, powers;
- enemy: hp, Block, intent, powers;
- decide how to play card.

Your admissible actions are:
- "play card <card_idx>" (when cards don't require a target and has sufficient energy)
- "play card <card_idx> on enemy <enemy_idx>" (when attacks or targeted skills and has sufficient energy)
- "end turn" (when you have no playable cards, no energy)

Examples:
- <action>play card 3 on enemy 1</action>
- <action>play card 2</action>
- <action>end turn</action>

You should first reason step-by-step about the current state, then think carefully which admissible action best advances the goal of winning the combat. 
This reasoning process MUST be enclosed within <think></think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action></action> tags.
"""


def make_state_prompt(state: dict) -> str:
    """Create structured state prompt for the current game state.
    
    Returns: state_text (user message content)
    """
    phase_info = state.get("phase_info", {})
    phase = phase_info.get("phase", "Unknown")
    
    
    lines = [f"[PHASE: {phase}]"]
    lines.append("")
    lines.append("=== CURRENT STATE ===")
    
    combat = state.get("combat", {})
    if combat:
        player = combat.get("player", {})
        if player:
            hp_str = state.get("hp", "N/A")
            lines.append("")
            lines.append("[PLAYER]")
            lines.append(f"  HP: {hp_str}")
            lines.append(f"  Block: {player.get('block', 0)}")
            lines.append(f"  Energy: {combat.get('energy', 0)}/{combat.get('max_energy', 3)}")
            
            powers = player.get("powers", [])
            if powers:
                power_str = ", ".join([f"{p.get('id', '?')}({p.get('amount', 0)})" for p in powers])
                lines.append(f"  Powers: {power_str}")
            
            hand = player.get("hand", [])
            if hand:
                lines.append("")
                lines.append("[HAND]")
                for i, card in enumerate(hand):
                    name = card.get("name", "Unknown")
                    cost = card.get("cost", 0)
                    dmg = card.get("base_damage") or 0
                    blk = card.get("base_block") or 0
                    
                    effect = ""
                    if name in ironclad_cards:
                        effect = ironclad_cards[name].get("effect", "")
                    
                    card_info = f"  card{i}: {name} (Cost: {cost})"

                    if effect:
                        card_info += f" | {effect[:40]}"
                    lines.append(card_info)
        
        enemies = combat.get("enemies", [])
        if enemies:
            lines.append("")
            lines.append("[ENEMIES]")
            for i, enemy in enumerate(enemies):
                enemy_info = f"  enemy{i}: {enemy.get('name', '?')}"
                hp_str = enemy.get('hp', '?')
                if '/' not in str(hp_str):
                    hp_str = f"{hp_str}/{enemy.get('max_hp', '?')}"
                enemy_info += f" | HP: {hp_str}"
                enemy_info += f" | Block: {enemy.get('block', 0)}"
                
                powers = enemy.get("powers", [])
                if powers:
                    power_str = ", ".join([f"{p.get('id', '?')}({p.get('amount', 0)})" for p in powers])
                    enemy_info += f" | Powers: {power_str}"
                
                lines.append(enemy_info)
                
                intent = enemy.get("intent", [])
                if intent:
                    intent_strs = []
                    for it in intent:
                        intent_type = it.get("type", "Unknown")
                        damage = it.get("damage", 0)
                        hits = it.get("hits", 1)
                        intent_strs.append(f"{intent_type}:{damage}x{hits}")
                    lines.append(f"    Intent: [{', '.join(intent_strs)}]")
    
    return "\n".join(lines)


def check_format_reward(response_text: str) -> float:
    """Check response format and return format reward.
    
    Format rewards:
    - Has <think> tags: +0.1
    - Has </think> tags: +0.1
    - Has <action> tags: +0.1
    - Has </action> tags: +0.1
    - Total maximum format reward: +0.4
    """
    reward = 0.0
    
    if re.search(r'<think>', response_text, re.IGNORECASE):
        reward += 0.1
    if re.search(r'</think>', response_text, re.IGNORECASE):
        reward += 0.1
    if re.search(r'<action>', response_text, re.IGNORECASE):
        reward += 0.1
    if re.search(r'</action>', response_text, re.IGNORECASE):
        reward += 0.1
    
    return reward

def parse_action(response_text: str) -> int:
    """Parse action from model response.
    
    Supports three action formats:
    - "end turn" -> 0
    - "play card N" -> N+1 (cards 1-10 map to actions 1-10)
    - "play card N on enemy M" -> (N) * 10 + M + 10 (e.g., card1 on enemy0 -> 11)
    
    Extracts action text from <action></action> tags first.
    Returns 0 if no valid action found.
    """
    action_match = re.search(r'<action>\s*(.*?)\s*</action>', response_text, re.IGNORECASE | re.DOTALL)
    
    if not action_match:
        return 0
    
    action_text = action_match.group(1).strip().lower()
    
    if not action_text:
        return 0
    
    if 'end turn' in action_text:
        return 0
    
    card_on_enemy = re.search(r'play\s+card\s+(\d+)\s+on\s+enemy\s+(\d+)', action_text)
    if card_on_enemy:
        card_num = int(card_on_enemy.group(1)) + 1
        enemy_num = int(card_on_enemy.group(2)) + 1
        action = (card_num) * 10 + enemy_num
        return min(max(action, 0), 156)
    
    play_card = re.search(r'play\s+card\s+(\d+)', action_text)
    if play_card:
        card_num = int(play_card.group(1)) + 1
        action = card_num
        return min(max(action, 0), 156)
    
    return 0