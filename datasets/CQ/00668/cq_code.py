import cadquery as cq

result = (
cq.Workplane("front")
.box(66, 10, 70)
.faces(">Y")
.workplane()
.move(0,  2)
.rect(47.1, 47., forConstruction = True)
.vertices()
.cboreHole(5, 7.5, 4)
.workplane()
.move(0, 2)
.cboreHole(28.2, 40, 2)
)
cq.exporters.export(result, 'GT.stl')